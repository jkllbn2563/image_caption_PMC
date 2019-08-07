import argparse
import os
import json
from pprint import pprint
from random import shuffle

parser = argparse.ArgumentParser()
parser.add_argument('--output_json', type=str, default='./my_data_ref.json',
                    help='directory for saving json file')
args = parser.parse_args()

# read line
with open('sam_new_dataset_reference.txt') as f:
    content = f.readlines()
content = [x.strip() for x in content]

# write json
json_images = []
caption_id = 0
image_id = 0
images_id_dict={}
for line in content:
    sentence = line[line.find(' ')+1:]
    if not (line.split()[0] in images_id_dict):
        json_images.append({'filename': line.split()[0], 'img_id': image_id,
            'filepath':'', 'split': 'train', 'cocoid': image_id, 'sentids': [],
            'sentences': []})
        images_id_dict[line.split()[0]] = image_id
        image_id += 1
    json_images[images_id_dict[line.split()[0]]]['sentids'].append(caption_id)
    if (sentence[-1]=='.'):
        tokens = sentence[:-1].split()
        raw = sentence+' '
    else:
        tokens = sentence.split()
        raw = sentence+'. '
    imgid = json_images[images_id_dict[line.split()[0]]]['img_id']
    caption_dict = {'tokens': tokens, 'raw': raw, 'imgid': imgid, 'sentid': caption_id}
    json_images[images_id_dict[line.split()[0]]]['sentences'].append(caption_dict)
    caption_id += 1
    
data = {'images': json_images, 'dataset':'coco'}
with open(args.output_json, 'w') as outfile:
    json.dump(data, outfile)
