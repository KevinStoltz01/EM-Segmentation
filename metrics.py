import glob
import numpy as np 
import os 
from PIL import Image
import re
import sys

ground_truth = "" 
predictions = ""
classes = 2

    
def Metrics(gt_dir_path, pred_dir_path, no_classes, thresh):
    '''
    A function that takes as input a directory of ground truth images and a directory of images predicted by a CNN. 
    Calculates average accuracy and mean intersection 

    gt_dir_path: file path to the directory containing ground truth images
    pred_dir_path: file path to the directory containing predicted images
    no_classes: number of prediction classes
    thresh: value at which to threshold predicted images 
    '''

    # set and sort the file paths for each image in the ground truth and prediction directories 
    gt_dir = glob.glob(os.path.join(gt_dir_path, "*"))
    gt_dir.sort()
    gt_names = os.listdir(gt_dir_path)
    pred_dir = glob.glob(os.path.join(pred_dir_path, "*"))
    pred_dir.sort()
    pred_names = os.listdir(pred_dir_path)

    # sanirty checks to ensure that ground truth and prediction directories contain the same number of files and have the same names, in order
    if len(gt_dir) != len(pred_dir):
            raise ValueError("Ground truth and predictions directories need to have the same number of files")
    if gt_names != pred_names:
            raise ValueError("Names for ground truth and prediction files must match exactly")

    # counters that will be used to calculate average accuracy and miou
    accuracy = 0
    miou = 0
    # calculates accuracy and miou for each image in the prediction directory
    for gt_file, pred_file in zip(gt_dir, pred_dir):
        # load ground truth array and normalize, load prediction image 
        gt = np.asarray(Image.open(gt_file))
        gt = gt/255
        pred = np.asarray(Image.open(pred_file))
    
        # the numpy arrays we loaded in are immutable so we need to make them writable 
        pred.setflags(write=1)
        # threshold the prediction array and flatten 
        pred[pred <= thresh] = 0
        pred[pred != 0] = 1
        gt = gt.flatten()
        pred = pred.flatten()

        # number of prediction classes
        ncl = float(no_classes)
        # number of black pixels in the ground truth image
        t0 = float(np.sum(np.where(gt == 0, 1, 0)*1))
        # number of white pixels in the ground truth image
        t1 = float(np.sum(np.where(gt == 1, 1, 0)*1))
        # number of correctly classified black pixels 
        n00 = float(np.sum(np.logical_and(pred == 0, gt == 0)*1))
        # number of correctly classified white pixels 
        n11 = float(np.sum(np.logical_and(pred == 1, gt == 1)*1))
        # number of false positives 
        n01 = float(np.sum(np.logical_and(pred == 1, gt == 0)*1))
         # number of false negatives
        n10 = float(np.sum(np.logical_and(pred == 0, gt == 1)*1))
        
        # accuracy calculation 
        acc = float((n00 + n11)/(t0 + t1))
        accuracy.append(acc)
        # miou calculation 
        if np.array_equal(gt, pred):
            miou += 1
        else:    
            mIOU = float((1/ncl) * ((n11/(t1 + n01)) + (n00/(t0 + n10))))
            miou += mIOU
    # average accuracy and mean miou calculations
    avg_miou = miou/len(gt_dir)
    avg_acc = accuracy/len(gt_dir)

    return avg_acc, avg_miou


def MetricsinMemory(gt, pred, no_classes, thresh=0.5):
    '''
    A function that takes in a ground truth image and a prediction image and calculates mean accuracy and intersection over union.
    These calculations are performed in memory, negating the need to save a directory of images before calculating metrics. 

    gt: ground truth image
    pred: predicted image from a trained CNN
    no_classes: the number of prediction classes
    thresh: value at which to threshold the prediction images
    '''
    # threshold predictions and flatten images to vectors 
    pred[pred <= thresh] = 0
    pred[pred != 0] = 1
    gt = gt.flatten()
    pred = pred.flatten()

    # Calculate pixel values for metrics calculations
    # number of prediction classes
    ncl = float(no_classes)
    # number of black pixels in ground truth image
    t0 = float(np.sum(np.where(gt == 0, 1, 0)*1))
    # number of white pixels in ground truth image
    t1 = float(np.sum(np.where(gt == 1, 1, 0)*1)) 
    # number of correctly classified black pixles
    n00 = float(np.sum(np.logical_and(pred == 0, gt == 0)*1)) 
    # number of correctly classified white pixels
    n11 = float(np.sum(np.logical_and(pred == 1, gt == 1)*1))
    # number of false positives 
    n01 = float(np.sum(np.logical_and(pred == 1, gt == 0)*1))
    # number of false negatives 
    n10 = float(np.sum(np.logical_and(pred == 0, gt == 1)*1))

    # accuracy calculation 
    acc = float((n00 + n11)/(t0 + t1))
    # miou caluclation 
    if np.array_equal(gt, pred):
        miou = 1
    else: 
        miou = float((1/ncl) * ((n11/(t1 + n01)) + (n00/(t0 + n10))))

    return acc, miou

    
def metrics_grid_search(gt_dir_path, pred_dir_path, no_classes):
    '''
    Implements a grid search over several threshold values and determines the best threshold (by mIOU metric) for a given dataset

    gt_dir_path: file path to ground truth directory
    pred_dir_path: file path to prediction directory
    no_classes: number of classes
    '''

    temp_acc = []
    temp_miou = []
    thresholds = [0.9, 0.8, 0.7, 0.6, 0.5]

    for threshold in thresholds:
        local_acc, local_miou = Metrics(gt_dir_path, pred_dir_path, 2, threshold)
        print(local_acc)
        temp_acc.append(local_acc)
        temp_miou.append(local_miou)

    temp_miou_argmax = np.argmax(temp_miou)
    # print("Accuracy: ", (temp_acc[int(temp_miou_argmax)]))
    # print("mIOU: ", (temp_miou[int(temp_miou_argmax)]))
    # print("Threshold: ", (thresholds[int(temp_miou_argmax)]))

    return temp_acc[int(temp_miou_argmax)], temp_miou[int(temp_miou_argmax)], thresholds[int(temp_miou_argmax)] 


def metrics_in_memory_grid_search(gt_in_memory, pred_in_memory, no_classes):
    temp_acc = []
    temp_miou = []
    thresholds = [0.9, 0.8, 0.7, 0.6, 0.5]

    for threshold in thresholds:
        local_acc, local_miou = MetricsinMemory(gt_in_memory, pred_in_memory, 2, threshold)
        temp_acc.append(local_acc)
        temp_miou.append(local_miou)
        del gt_in_memory         
        print("threshold - {}, accuracy - {}, miou - {}".format(threshold, local_acc, local_miou))

    temp_miou_argmax = np.argmax(temp_miou)
    # print("temp miou argmax - {}".format(temp_miou_argmax))


    return temp_acc[int(temp_miou_argmax)], temp_miou[int(temp_miou_argmax)], thresholds[int(temp_miou_argmax)] 


if __name__ == '__main__':
    Metrics(ground_truth, predictions, classes, 0.9)
