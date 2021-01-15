from PIL import Image
from keras.models import model_from_json
import os
import utils
import numpy as np
from pathlib import Path
#import cv2
import argparse
import tensorflow as tf

from utils import integrate, integrateChunk, chopup


parser = argparse.ArgumentParser()
parser.add_argument('--out_dir', type=str, default="../results/{}")
parser.add_argument('--f_model_dir', type=str, default="../pretrained")
parser.add_argument('--f_model', type=str, default="../pretrained/{}_best.json")
parser.add_argument('--f_weights', type=str, default="../pretrained/{}_best.h5")
parser.add_argument('--f_dataset', type=str, default="DRIVE")
parser.add_argument('--spec', type=str, default="1.tmp")

HACKMODE = False


args = parser.parse_args()

with tf.device('/gpu:0'):
    out_dir = args.out_dir
    dataset = args.f_dataset
    spec = args.spec

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
    elif dataset == 'CHASE' or dataset == 'CHASEDB1':
        img_size = (1024, 1024)

    # ori_shape=(1,584,565) if dataset=="DRIVE" else (1,605,700)  # batchsize=1
    if dataset=="DRIVE":
        ori_shape=(1,584,565)
    elif dataset=="STARE":
        ori_shape=(1,605,700)
    elif dataset=="miniDROPS":
        ori_shape=(1,480,640)
    elif dataset == 'CHASE' or dataset == 'CHASEDB1':
        ori_shape = (1, 960, 999)

    fundus_files = []
    mask_files = []
    with open(spec, "r") as sp:
        l = sp.readlines()
        fundus_files = [v.strip() for (i, v) in enumerate(l) if i % 2 == 0]
        mask_files = [v.strip() for (i, v) in enumerate(l) if i % 2 == 1]

    #fundus_files=utils.all_files_under(fundus_dir.format(dataset))
    #mask_files=utils.all_files_under(mask_dir.format(dataset), extension="png")


    for index,fundus_file in enumerate(fundus_files):
        print(f'Processing {fundus_file}')
        img=utils.imagefiles2arrs([fundus_file])
        mask=utils.imagefiles2arrs([mask_files[index]])
        if not HACKMODE:
            dx = 1 if img.shape[1] > ori_shape[1] else 0
            dy = 1 if img.shape[2] > ori_shape[2] else 0
            img = np.array([img[0,dx:,dy:,:]])
        else:
            ori_shape=(1,605,700)
            img_size=(720,720)
            #print(f"I'm converting an image that's {img.shape} to one that's {ori_shape}")
            #img = np.array([cv2.resize(img[0], (ori_shape[2], ori_shape[1]))])
            #mask = np.array([cv2.resize(mask[0], (ori_shape[2], ori_shape[1]))])
        # z score with mean, std (batchsize=1)
        mean=np.mean(img[0,...][mask[0,...] > 0.99],axis=0)
        std=np.std(img[0,...][mask[0,...] > 0.99],axis=0)
        img[0,...]=(img[0,...]-mean)/std
        
        # run inference
        padded_img=utils.pad_imgs(img, img_size)
        bits = chopup(padded_img, 128)
        bits2 = bits[:, 0, :, :, :]
        #print(f'Input: {bits2.shape}')
        #vessel_img_bits = np.zeros((64, 1024, 1024, 1))
        #for i in range(4):
            #vessel_img_bits[i*16:(i+1)*16,...]=model.predict(bits2[i*16:(i+1)*16,...],batch_size=16)*255
        vessel_img_bits=model.predict(bits2,batch_size=64)*255
        #print(f'Output: {vessel_img_bits.shape}')
        vessel_img = np.asarray([integrateChunk(vessel_img_bits)])
        #print(f'Integrated vessel: {vessel_img.shape}')
        cropped_vessel=utils.crop_to_original(vessel_img[...,0], ori_shape)
        #print(f'Cropped vessel: {cropped_vessel.shape}')
        final_result=utils.remain_in_mask(cropped_vessel[0,...], mask[0,...])
        if HACKMODE:
            ori_shape = (1, 960, 999)
            #final_result = cv2.resize(final_result, (ori_shape[2], ori_shape[1]))
        outfile = Path(out_dir.format(dataset))
        outfile = outfile / 'probability_maps' / f'{Path(fundus_file).stem}.png'
        Image.fromarray(final_result.astype(np.uint8)).save(outfile)
