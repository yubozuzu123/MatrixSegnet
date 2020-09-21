#!/usr/bin/env python
import os
import json
import torch
import pprint
import argparse
import importlib
import numpy as np
from torchvision import transforms
from tensorboardX import SummaryWriter
from models.data_loader import test_dataloader
import matplotlib
matplotlib.use("Agg")
#from lib.config import cfg, args
from config import system_configs
from nnet.py_factory import NetworkFactory
from db.datasets import datasets
import cv2
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time


torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def iou(y_true, y_pred, n_class):
    # IOU = TP/(TP+FN+FP)
    IOU = []
    '''
    for c in range(n_class):
        TP = np.sum((y_true == c) & (y_pred == c))
        FP = np.sum((y_true != c) & (y_pred == c))
        FN = np.sum((y_true == c) & (y_pred != c))

        n = TP
        d = float(TP + FP + FN + 1e-12)

        iou = np.divide(n, d)
        IOU.append(iou)
    '''
    #recall=tp/(tp+fn)
    #precision=tp/(tp+fp)
    
  
          
    return iou_single,recall,precision, F1

def parse_args():
    parser = argparse.ArgumentParser(description="Test MatrixNet")
    parser.add_argument("cfg_file", help="config file", type=str)
    parser.add_argument("--testiter", dest="testiter",
                        help="test at iteration i",
                        default=None, type=int)
    parser.add_argument("--split", dest="split",
                        help="which split to use",
                        default="validation", type=str)
    parser.add_argument("--suffix", dest="suffix", default=None, type=str)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    return args

def make_dirs(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

def test(): 
    result_dir = "/data3/building_dataset/whu_building_Dataset/cropped images/test/result/"

    print("building neural network...")
    nnet =  NetworkFactory(test_dataloader)
    print("loading parameters...")
    test_model_dir="/home/boyu/matrixnet-master/MatrixNetAnchors_whu/nnet/MatrixNetAnchors/MatrixNetAnchors_20481.pkl"
    nnet.load_params(20481)
    nnet.cuda()
    
    test_image_dir='/data3/building_dataset/whu_building_Dataset/cropped images/test/image/'
    
    num_img=len(os.listdir(test_image_dir))
    classnum = 2
    iou_totoal_mean=0.0
    recall_mean=0.0
    precision_mean=0.0
    F1_mean=0.0
    img_divide=0
    tp_builging_total=0
    fp_building_total=0
    fn_buildling_total=0
    time_sum=0
    res=[]
    for index in range(num_img):
       img_name = os.listdir(test_image_dir)[index]
       imgA = cv2.imread('/data3/building_dataset/whu_building_Dataset/cropped images/test/image/'+img_name)
       imgA = cv2.resize(imgA, (512, 512))
       imgA = transform(imgA)
       imgA = imgA.cuda()
       imgA = imgA.unsqueeze(0)
       start=time.time()
       output=nnet.test(imgA)
       #output = torch.sigmoid(output)
       end=time.time()
       res.append(end-start)

       output_np = output.cpu().detach().numpy().copy() 
       output_np = np.argmax(output_np, axis=1)
       output_array=np.squeeze(output_np[0, ...])
       output_array=1-output_array
       output_array_save=255*output_array
       result_path=result_dir+img_name[:-4]+'.png'
       cv2.imwrite(result_path,output_array_save)
      
       
       gt = cv2.imread('/data3/building_dataset/whu_building_Dataset/cropped images/test/label/'+img_name,0)
       gt = cv2.resize(gt, (512, 512))
       gt = gt/255
       
       #miou=iou_mean(output_np,gt)
       #print("mean_iou is: {}".format(miou.item()))
       img_divide=img_divide+1
       tp_builging=np.sum((gt == 1) & (output_array == 1))
       fp_building= np.sum((gt != 1) & (output_array == 1))
       fn_buildling =np.sum((gt == 1) & (output_array != 1))
       
       tp_builging_total=tp_builging_total+tp_builging
       fp_building_total=fp_building_total+fp_building
       fn_buildling_total=fn_buildling_total+fn_buildling
    
       recall =  np.divide(tp_builging, float(tp_builging+fn_buildling))
       precision = np.divide(tp_builging, float(tp_builging+fp_building))   
       F1 =np.divide(2*recall*precision,(recall+precision))
       iou_single=np.divide(tp_builging,float(tp_builging+fp_building+fn_buildling))
       if tp_builging == 0:
         iou_single=0.0
         recall=0.0
         precision=0.0
         F1=0.0
         img_divide=img_divide-1
       #print("iou=%f"%iou_single)
      # print("recall=%f"%recall)
      # print("precision=%f"%precision)
      # print("F1_mean=%f"%F1)
       iou_totoal_mean=iou_totoal_mean+iou_single
       recall_mean=recall_mean+recall
       precision_mean=precision_mean+precision
       F1_mean=F1_mean+F1
       
          
    iou_totoal_mean=np.divide(iou_totoal_mean, float(img_divide))
    recall_mean=np.divide(recall_mean, float(img_divide))
    precision_mean=np.divide(precision_mean, float(img_divide))
    F1_mean=np.divide(F1_mean, float(img_divide))
    print("iou=%f"%iou_totoal_mean)
    print("recall=%f"%recall_mean)
    print("precision=%f"%precision_mean)
    print("F1_mean=%f"%F1_mean)
    
    iou_total=np.divide(tp_builging_total,float(tp_builging_total+fp_building_total+fn_buildling_total))
    recall_total =  np.divide(tp_builging_total, float(tp_builging_total+fn_buildling_total))
    precision_total = np.divide(tp_builging_total, float(tp_builging_total+fp_building_total))   
    F1_total =np.divide(2*recall_total*precision_total,(recall_total+precision_total))
    print("iou_total=%f"%iou_total)
    print("recall_total=%f"%recall_total)
    print("precision_total=%f"%precision_total)
    print("F1_mean_total=%f"%F1_total)
    
    for i in res:
       time_sum +=i
    print("FPS: %f"%(1.0/(time_sum/len(res))))
       



if __name__ == "__main__":
    cfg_file = "/home/boyu/matrixnet-master/config/matrixsegnet.json"
    with open(cfg_file, "r") as f:
        configs = json.load(f)
    system_configs.update_config(configs["system"])

    test()
    
