from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#prepo_feats
import os
import sys
ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)

import json
import argparse
import h5py
from random import shuffle, seed

import numpy as np
import torch
from torch.autograd import Variable
import skimage.io

from torchvision import transforms as trn
preprocess = trn.Compose([
        #trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

from misc.resnet_utils import myResnet
import misc.resnet as resnet

#terence edit start
def prepo_feats_init(params_p):
    global net, my_resnet, imgs, N, params
    params = params_p
    #terence edit end
    net = getattr(resnet, params['model'])()
    net.load_state_dict(torch.load(os.path.join(params['model_root'],params['model']+'.pth')))
    my_resnet = myResnet(net)
    my_resnet.cuda()
    my_resnet.eval()
    imgs = json.load(open(params['input_json_original'], 'r'))
    imgs = imgs['images']
    N = len(imgs)
    seed(123) # make reproducible
#terence edit start
def terence_get_fc_att(index):
    img = [x for x in imgs if x['cocoid']==int(index)]
    assert len(img)>0
    img = img[0]
    #terence edit end
    # load the image
    I = skimage.io.imread(os.path.join(params['images_root'], img['filepath'], img['filename']))
    # handle grayscale input images
    if len(I.shape) == 2:
        I = I[:,:,np.newaxis]
        I = np.concatenate((I,I,I), axis=2)     
        
    I = I.astype('float32')/255.0
    I = torch.from_numpy(I.transpose([2,0,1])).cuda()
    with torch.no_grad():   
        I = Variable(preprocess(I))
        tmp_fc, tmp_att = my_resnet(I, params['att_size'])
    #print (tmp_fc.shape)
    #print (tmp_att.shape)
    return tmp_fc, tmp_att 
