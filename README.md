# image_caption_PMC
圖片放在 /sam_new_dataset/images
label 在 sam_new_dataset_reference_with_val.txt

python sam_new_dataset_json_file_with_val.py --output_json sam_new_dataset_reference.json 產生 sam_new_dataset_reference.json #產生 cocotalk.json格式的檔案

python scripts/prepro_labels.py --input_json sam_new_dataset/sam_new_dataset_reference.json --output_json sam_new_dataset/cocotalk_sam_new_dataset_reference.json --output_h5 sam_new_dataset/cocotalk_sam_new_dataset_reference --word_count_threshold 0
#產生 cocotalk_sam_new_dataset_reference.json  cocotalk_sam_new_dataset_reference_label.h5

python scripts/prepro_labels_assign_vocab.py --input_json sam_new_dataset/sam_new_dataset_reference.json --output_json sam_new_dataset/cocotalk_sam_new_dataset_reference.json --output_h5 sam_new_dataset/cocotalk_sam_new_dataset_reference --
word_count_threshold 0

#產生 cocotalk_sam_new_dataset_reference.json  cocotalk_sam_new_dataset_reference_label.h5

export NEW_DATASET_IMAGE_DIR=$HOME/code/python/pytorch/image_caption/ruotianluo_image_captioning_pytorch/sam_new_dataset/images

python scripts/prepro_feats.py --input_json sam_new_dataset/sam_new_dataset_reference.json --output_dir sam_new_dataset/cocotalk_sam_new_dataset --images_root $NEW_DATASET_IMAGE_DIR
#產生 coco_talk_att coco_talk_fc

python fine_tune_with_val.py --id st --caption_model topdown --input_json sam_new_dataset/cocotalk_sam_new_dataset_reference.json --input_fc_dir sam_new_dataset/cocotalk_sam_new_dataset_fc --input_att_dir sam_new_dataset/cocotalk_sam_new_dataset_att --input_label_h5 sam_new_dataset/cocotalk_sam_new_dataset_reference_label.h5 --batch_size 4 --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --checkpoint_path log_st --save_checkpoint_every 4 --val_images_use 5000 --max_epochs 40

###
append customer image on coco
###
ALT+F2: xkill

python fine_tune.py --output_json coco_append.json 產生 coco_append.json #產生 cocotalk.json格式的檔案 remember dataset_coco.json in bak.json dir

python scripts/prepro_labels.py --input_json sam_new_dataset/sam_new_dataset_reference.json --output_json sam_new_dataset/cocotalk_sam_new_dataset_reference.json --output_h5 sam_new_dataset/cocotalk_sam_new_dataset_reference --word_count_threshold 0

###產生 cocotalk_sam_new_dataset_reference.json  cocotalk_sam_new_dataset_reference_label.h5

###export NEW_DATASET_IMAGE_DIR=/media/jkllbn2563/20ea5ca3-ac85-48bf-831d-06ebe29cef77/data/train_val2014

sudo python scripts/prepro_feats.py --input_json /home/jkllbn2563/catkin_ws/src/scorpio_v2/image_caption_PMC/src/sam_new_dataset/coco_append.json --output_dir /home/jkllbn2563/catkin_ws/src/scorpio_v2/image_caption_PMC/src/sam_new_dataset/cocotalk_sam_new_dataset --images_root

$NEW_DATASET_IMAGE_DIR      at ~/code/python/pytorch/self-critical.pytorch

####產生 coco_talk_att coco_talk_fc

~/code/python/pytorch/self-critical.pytorch$ python train.py --cfg configs/updown.yml --id updown


＃＃＃強化學習

python scripts/prepro_ngrams.py --input_json data/coco_append.json --dict_json /home/jkllbn2563/catkin_ws/src/scorpio_v2/image_caption_PMC/src/sam_new_dataset/cocotalk_sam_new_dataset_reference.json --output_pkl data/coco-train --split train

bash scripts/copy_model.sh fc fc_rl

python train.py --cfg configs/fc_rl.yml --id fc_rl
