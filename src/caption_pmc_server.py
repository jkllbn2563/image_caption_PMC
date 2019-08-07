#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from six.moves import cPickle
import models
import argparse
import misc.utils as utils
import torch
from torch.autograd import Variable
import skimage
from torchvision import transforms as trn
from misc.resnet_utils import myResnet
import misc.resnet

import rospy
from caption_pkg.srv import *
from cv_bridge import CvBridge,CvBridgeError
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
bridge = CvBridge()

# Input arguments and options
parser = argparse.ArgumentParser()
# Input paths
parser.add_argument('--model', type=str, default='log_st/model-best.pth',
                    help='path to model to evaluate')
parser.add_argument('--image', type=str, default='',
                    help='test image')
parser.add_argument('--cnn_model', type=str,  default='resnet101',
                    help='resnet101, resnet152')
parser.add_argument('--infos_path', type=str, default='log_st/infos_st-best.pkl',
                    help='path to infos to evaluate')
#parser.add_argument('--infos_path', type=str, default='no_finetune_pre-trained_models/topdown/infos_td-best.pkl',
#                    help='path to infos to evaluate')
# Basic options
parser.add_argument('--batch_size', type=int, default=0,
                    help='if > 0 then overrule, otherwise load from checkpoint.')
parser.add_argument('--num_images', type=int, default=-1,
                    help='how many images to use when periodically evaluating the loss? (-1 = all)')
parser.add_argument('--language_eval', type=int, default=0,
                    help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
parser.add_argument('--dump_images', type=int, default=1,
                    help='Dump images into vis/imgs folder for vis? (1=yes,0=no)')
parser.add_argument('--dump_json', type=int, default=1,
                    help='Dump json with predictions into vis folder? (1=yes,0=no)')
parser.add_argument('--dump_path', type=int, default=0,
                    help='Write image paths along with predictions into vis json? (1=yes,0=no)')

# Sampling options
parser.add_argument('--sample_max', type=int, default=1,
                    help='1 = sample argmax words. 0 = sample from distributions.')
parser.add_argument('--beam_size', type=int, default=2,
                    help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature when sampling from distributions (i.e. when sample_max = 0). Lower = "safer" predictions.')
# For evaluation on a folder of images:
parser.add_argument('--image_folder', type=str, default='silicon_test_images',
                    help='If this is nonempty then will predict on the images in this folder path')
parser.add_argument('--image_root', type=str, default='',
                    help='In case the image paths have to be preprended with a root path to an image folder')
# For evaluation on MSCOCO images from some split:
parser.add_argument('--input_fc_dir', type=str, default='',
                    help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_att_dir', type=str, default='',
                    help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_label_h5', type=str, default='',
                    help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_json', type=str, default='',
                    help='path to the json file containing additional info and vocab. empty = fetch from model checkpoint.')
parser.add_argument('--split', type=str, default='test',
                    help='if running on MSCOCO images, which split to use: val|test|train')
parser.add_argument('--coco_json', type=str, default='',
                    help='if nonempty then use this file in DataLoaderRaw (see docs there). Used only in MSCOCO test evaluation, where we have a specific json file of only test set images.')
# misc
parser.add_argument('--id', type=str, default='',
                    help='an id identifying this run/job. used only if language_eval = 1 for appending to intermediate files')

opt = parser.parse_args()
use_cuda = False
if torch.cuda.is_available():
    use_cuda = True


preprocess = trn.Compose([
        #trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load infos
with open(opt.infos_path, 'rb') as f:
    infos = cPickle.load(f)


# override and collect parameters
if len(opt.input_fc_dir) == 0:
    opt.input_fc_dir = infos['opt'].input_fc_dir
    opt.input_att_dir = infos['opt'].input_att_dir
    opt.input_label_h5 = infos['opt'].input_label_h5
if len(opt.input_json) == 0:
    opt.input_json = infos['opt'].input_json
if opt.batch_size == 0:
    opt.batch_size = infos['opt'].batch_size
if len(opt.id) == 0:
    opt.id = infos['opt'].id
ignore = ["id", "batch_size", "beam_size", "start_from", "language_eval"]
for k in vars(infos['opt']).keys():
    if k not in ignore:
        if k in vars(opt):
            assert vars(opt)[k] == vars(infos['opt'])[k], k + ' option not consistent'
        else:
            vars(opt).update({k: vars(infos['opt'])[k]}) # copy over options from model

vocab = infos['vocab'] # ix -> word mapping


model = models.setup(opt)
if use_cuda == False:
    model.load_state_dict(torch.load(opt.model,map_location='cpu'))
else:
    model.load_state_dict(torch.load(opt.model))
    model.cuda()
model.eval()

my_resnet = getattr(misc.resnet, 'resnet101')()
my_resnet.load_state_dict(torch.load('./data/imagenet_weights/'+'resnet101'+'.pth'))
my_resnet = myResnet(my_resnet)
if use_cuda:
    my_resnet.cuda()
my_resnet.eval()
batch_size = 1
info_struct = {}
info_struct['id'] = 0
info_struct['file_path'] = ''
infos = []
infos.append(info_struct)
data = {}
data['bounds'] = {'it_pos_now': 0, 'it_max': 1, 'wrapped': True}
data['infos'] = infos



def image_captioning_body(img):

    #img = skimage.io.imread(opt.image)
    #img = skimage.io.imread('silicon_test_images/cellphone.jpg')
    if (img.shape[0]==2):
        img = img[0]
    print (img.shape)


    fc_batch = np.ndarray((batch_size, 2048), dtype = 'float32')
    att_batch = np.ndarray((batch_size, 14, 14, 2048), dtype = 'float32')
    if len(img.shape) == 2:
        img = img[:,:,np.newaxis]
        img = np.concatenate((img, img, img), axis=2)

    img = img.astype('float32')/255.0
    img = torch.from_numpy(img.transpose([2, 0, 1]))
    '''
    if use_cuda==False:
        img = torch.from_numpy(img.transpose([2, 0, 1]))
    else:
        img = torch.from_numpy(img.transpose([2, 0, 1])).cuda()
    '''
    with torch.no_grad():
        img = Variable(preprocess(img))
        if use_cuda==True:
            img = img.cuda()
        tmp_fc, tmp_att = my_resnet(img)

    fc_batch[0] = tmp_fc.data.cpu().float().numpy()
    att_batch[0] = tmp_att.data.cpu().float().numpy()
    data['fc_feats'] = fc_batch
    data['att_feats'] = att_batch

    tmp = [data['fc_feats'][np.arange(batch_size)], 
        data['att_feats'][np.arange(batch_size)]]

    with torch.no_grad():
        if use_cuda:
            tmp = [Variable(torch.from_numpy(_)).cuda() for _ in tmp]
        else:
            tmp = [Variable(torch.from_numpy(_)) for _ in tmp]
        fc_feats, att_feats = tmp
        # forward the model to also get generated samples for each image
        seq, _ = model.sample(fc_feats, att_feats, vars(opt))

    seq = seq.cpu().numpy()
    sents = utils.decode_sequence(vocab, seq)
    print (sents)
    return sents[0]

def handle_function(req):
    print ('start to handle image captioning service')
    print (req.robot_view.height)
    cv_image = bridge.imgmsg_to_cv2(req.robot_view, desired_encoding="passthrough")
    #print (cv_image.shape)
    
   # cv2.imshow('frame',cv_image)
   # cv2.waitKey(0)
   # cv2.destroyAllWindows()
    
    sents = image_captioning_body(cv_image)

    return imageCaptioningResponse(sents)


rospy.init_node('caption_server',anonymous=True)
s=rospy.Service('image_caption',imageCaptioning,handle_function)
rospy.loginfo('Ready to caption')
rospy.spin()    
