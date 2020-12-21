from PIL import Image
from keras.models import model_from_json
import os
import utils
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--fundus_dir', type=str, default="../data/{}/test/images/")
parser.add_argument('--mask_dir', type=str, default="../data/{}/test/mask/")
parser.add_argument('--out_dir', type=str, default="../inference_outputs/{}")
parser.add_argument('--f_model_dir', type=str, default="../pretrained")
parser.add_argument('--f_model', type=str, default="../pretrained/{}_best.json")
parser.add_argument('--f_weights', type=str, default="../pretrained/{}_best.h5")
parser.add_argument('--f_dataset', type=str, default="DRIVE")

args = parser.parse_args()

fundus_dir = args.fundus_dir
mask_dir = args.mask_dir
out_dir = args.out_dir
dataset = args.f_dataset

if args.f_model is not None:
    f_model = args.f_model
    print(f_model)
if args.f_weights is not None:
    f_weights = args.f_weights


# make directory
if not os.path.isdir(out_dir.format(dataset)):
    os.makedirs(out_dir.format(dataset))

# load the model and weights
with open(f_model.format(dataset), 'r') as f:
    model=model_from_json(f.read())
model.load_weights(f_weights.format(dataset))

# iterate all images
# img_size=(640,640) if dataset=="DRIVE" else (720,720)
if dataset=="DRIVE":
    img_size=(640,640) 
elif dataset=="STARE":
    img_size=(720,720)
elif dataset=="miniDROPS":
    img_size=(640,640)

# ori_shape=(1,584,565) if dataset=="DRIVE" else (1,605,700)  # batchsize=1
if dataset=="DRIVE":
    ori_shape=(1,584,565)
elif dataset=="STARE":
    ori_shape=(1,605,700)
elif dataset=="miniDROPS":
    ori_shape=(1,480,640)

fundus_files=utils.all_files_under(fundus_dir.format(dataset))
mask_files=utils.all_files_under(mask_dir.format(dataset))
for index,fundus_file in enumerate(fundus_files):
    print("processing {}...".format(fundus_file))
    # load imgs
    img=utils.imagefiles2arrs([fundus_file])
    mask=utils.imagefiles2arrs([mask_files[index]])
    
    # z score with mean, std (batchsize=1)
    mean=np.mean(img[0,...][mask[0,...] == 255.0],axis=0)
    std=np.std(img[0,...][mask[0,...] == 255.0],axis=0)
    img[0,...]=(img[0,...]-mean)/std
    
    # run inference
    padded_img=utils.pad_imgs(img, img_size)
    vessel_img=model.predict(padded_img,batch_size=1)*255
    cropped_vessel=utils.crop_to_original(vessel_img[...,0], ori_shape)
    final_result=utils.remain_in_mask(cropped_vessel[0,...], mask[0,...])
    Image.fromarray(final_result.astype(np.uint8)).save(os.path.join(out_dir.format(dataset),os.path.basename(fundus_file)))
