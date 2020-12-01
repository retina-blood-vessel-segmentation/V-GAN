import utils
import os
from PIL import Image
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics.classification import confusion_matrix

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--comparison_out', type=str, default="../evaluation/{}/comparison/{}")
parser.add_argument('--vessels_out', type=str, default="../evaluation/{}/vessels/{}")
parser.add_argument('--curves_out', type=str, default="../evaluation/{}/measures")
parser.add_argument('--testdata', type=str, default="../data/{}/test/images")
parser.add_argument('--result_dir', type=str, default="../results")
parser.add_argument('--mask_dir', type=str, default="mask")

args = parser.parse_args()

comparison_out = args.comparison_out
vessels_out = args.vessels_out
curves_out = args.curves_out
testdata = args.testdata
result_dir = args.result_dir
mask_dir = args.mask_dir


# set output directories
# comparison_out="../evaluation/{}/comparison/{}"
# vessels_out="../evaluation/{}/vessels/{}"
# curves_out="../evaluation/{}/measures"
# testdata="../data/{}/test/images"

# draw 
# result_dir="../results"
datasets=utils.all_files_under(result_dir)
for dataset in datasets:
    all_results=utils.all_files_under(dataset)
    # mask_dir=os.path.join(dataset,"mask")             # u originalnom kodu je ovo otkomentarisano
    masks=utils.load_images_under_dir(mask_dir)/255
    gt_dir=os.path.join(dataset,"1st_manual")
    gt_vessels=utils.load_images_under_dir(gt_dir)/255
    
    # collect results from all methods
    methods=[]
    fprs, tprs, precs, recalls=[],[],[],[]
    for result in all_results:
        if ("mask" not in result):    #skip mask and ground truth
            # get pixels inside the field of view in fundus images
            pred_vessels=utils.load_images_under_dir(result)/255
            gt_vessels_in_mask, pred_vessels_in_mask = utils.pixel_values_in_mask(gt_vessels, pred_vessels , masks)
             
            # visualize results
            if "V-GAN" in result or "DRIU" in result or "1st_manual" in result:
                test_dir=testdata.format(os.path.basename(dataset))
                ori_imgs=utils.load_images_under_dir(test_dir)
                vessels_dir=vessels_out.format(os.path.basename(dataset),os.path.basename(result))
                filenames=utils.all_files_under(result)
                if not os.path.isdir(vessels_dir):
                    os.makedirs(vessels_dir)   
                for index in range(gt_vessels.shape[0]):
#                     thresholded_vessel=utils.threshold_by_f1(np.expand_dims(gt_vessels[index,...], axis=0),
#                                                                   np.expand_dims(pred_vessels[index,...], axis=0),
#                                                                   np.expand_dims(masks[index,...], axis=0), 
#                                                                   flatten=False)*255
                    thresholded_vessel=utils.threshold_by_otsu(np.expand_dims(pred_vessels[index,...], axis=0),
                                                                  np.expand_dims(masks[index,...], axis=0), 
                                                                  flatten=False)*255
                    ori_imgs[index,...][np.squeeze(thresholded_vessel, axis=0)==0]=(0,0,0)
#                     ori_imgs[index,...]*=np.tile(np.expand_dims(pred_vessels[index,...], axis=3), (1,1,3))
                    Image.fromarray(ori_imgs[index,...].astype(np.uint8)).save(os.path.join(vessels_dir,os.path.basename(filenames[index])))
                
                # compare with the ground truth
                comp_dir=comparison_out.format(os.path.basename(dataset),os.path.basename(result))
                if not os.path.isdir(comp_dir):
                    os.makedirs(comp_dir)
                dice_list=[]
                for index in range(gt_vessels.shape[0]):
                    diff_map, dice_coeff=utils.difference_map(gt_vessels[index,...], pred_vessels[index,...], masks[index,...])
                    dice_list.append(dice_coeff)
                    Image.fromarray(diff_map.astype(np.uint8)).save(os.path.join(comp_dir,os.path.basename(filenames[index])))
#                 print "indices of best dice coeff : {}".format(sorted(range(len(dice_list)),key=lambda k: dice_list[k]))
             
            # skip the ground truth
            if "1st_manual" not in result:
                # print metrics
                print "-- {} --".format(os.path.basename(result))
                print "dice coefficient : {}".format(utils.dice_coefficient(gt_vessels,pred_vessels, masks))
                print "f1 score : {}, accuracy : {}, sensitivity : {}, specificity : {}".format(*utils.misc_measures_evaluation(gt_vessels,pred_vessels, masks))
 
                # compute false positive rate, true positive graph
                method=os.path.basename(result)
                methods.append(method)
                if method=='CRFs' or method=='2nd_manual':
                    cm=confusion_matrix(gt_vessels_in_mask, pred_vessels_in_mask)
                    fpr=1-1.*cm[0,0]/(cm[0,1]+cm[0,0])
                    tpr=1.*cm[1,1]/(cm[1,0]+cm[1,1])
                    prec=1.*cm[1,1]/(cm[0,1]+cm[1,1])
                    recall=tpr
                    if method=='2nd_manual':
                        human_op_pts_roc, human_op_pts_pr = utils.operating_pts_human_experts(gt_vessels, pred_vessels, masks)
                else:
                    fpr, tpr, _ = roc_curve(gt_vessels_in_mask, pred_vessels_in_mask)
                    prec, recall, _ = precision_recall_curve(gt_vessels_in_mask, pred_vessels_in_mask)
                fprs.append(fpr)
                tprs.append(tpr)
                precs.append(prec)
                recalls.append(recall)
               
    # save plots of ROC and PR curves
    curve_dir=curves_out.format(os.path.basename(dataset))
    if not os.path.isdir(curve_dir):
        os.makedirs(curve_dir)
    
    utils.plot_AUC_ROC(fprs, tprs, methods, curve_dir, human_op_pts_roc)
    utils.plot_AUC_PR(precs, recalls, methods, curve_dir, human_op_pts_pr)    
