########train.py
from datetime import datetime
import json
import torch
import numpy as np
import queue
import pprint
import random
import argparse
import importlib
import threading
import traceback
import os

import cv2
from tqdm import tqdm
from utils import stdout_to_tqdm
from config import system_configs
from tqdm import tqdm
from utils import stdout_to_tqdm
from config import system_configs
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
from nnet.py_factory import NetworkFactory
from torch.multiprocessing import Process, Queue, Pool
 
from models.data_loader import train_dataloader,test_dataloader
from db.datasets import datasets


from torchvision import transforms
from tensorboardX import SummaryWriter
import matplotlib
matplotlib.use("Agg")
import torch.nn.functional as F
import time

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
def test(): 
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
       output_np = np.argmin(output_np, axis=1)
       output_array=np.squeeze(output_np[0, ...])
               
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
       
def parse_args():
    parser = argparse.ArgumentParser(description="Train MatrixNets")
    parser.add_argument("cfg_file", help="config file", type=str)
    parser.add_argument("--iter", dest="start_iter",
                        help="train at iteration i",
                        default=0, type=int)
    parser.add_argument("--threads", dest="threads", default=4, type=int)

    args = parser.parse_args()
    return args

def change_feature(check_point, num_class=2):

  	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  	check_point = torch.load(check_point, map_location=device)
  	
  	import collections
  	dicts = collections.OrderedDict()
  	for k, value in check_point.items():
  		if k == "module.net.subnet_anchors_heats.subnet_output.weight":
  			value = torch.ones(num_class, 256,3,3)
  		if k == "module.net.subnet_anchors_heats.subnet_output.bias":
  			value = torch.ones(num_class)
  		dicts[k] = value
  	torch.save(dicts, "./MatrixNetAnchorsResnet50_48LayerRange_640isize/nnet/MatrixNetAnchors/MatrixNetAnchors_50_modified.pkl")

def train(start_iter=20150):
    learning_rate = system_configs.learning_rate
    max_iteration = system_configs.max_iter
    pretrained_model = system_configs.pretrain
    snapshot = system_configs.snapshot
    val_iter = system_configs.val_iter
    display = system_configs.display
    decay_rate = system_configs.decay_rate
    stepsize = system_configs.stepsize
    
    #vis = visdom.Visdom()
 
  #  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("building model...")
    nnet = NetworkFactory(train_dataloader)
    #nnet = nnet.cuda()
    #nnet = nn.DataParallel(nnet).cuda() 
    if pretrained_model is not None:
        if not os.path.exists(pretrained_model):
            raise ValueError("pretrained model does not exist")
        print("loading from pretrained model")
        change_feature(pretrained_model)
        nnet.load_pretrained_params("./MatrixNetAnchorsResnet50_48LayerRange_640isize/nnet/MatrixNetAnchors/MatrixNetAnchors_50_modified.pkl")
        #params = torch.load(pretrained_model)
        #nnet.load_state_dict({k.replace('module.',''):v for k,v in params['state_dict'].items()})


    if start_iter:
        #learning_rate /= (decay_rate ** (start_iter // stepsize))
        #print(learning_rate)
        nnet.load_params(start_iter)
        nnet.set_lr(learning_rate)
        print("training starts from iteration {} with learning_rate {}".format(start_iter + 1, learning_rate))
    else:
        nnet.set_lr(learning_rate)
    
    print("training start...")
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nnet.cuda()
    with stdout_to_tqdm() as save_stdout:
        for iteration in tqdm(range(start_iter + 1, max_iteration + 1), file=save_stdout, ncols=80):
            loss_total=0
            nnet.train_mode()
            for index, (ls1,ls_msk) in enumerate(test_dataloader):
                training_loss = nnet.train(ls1,ls_msk)
                #print(training_loss)
                loss_total=loss_total+training_loss

            test_loss = 0
            nnet.eval_mode()
            with torch.no_grad():
                for index, (ls1, ls_msk) in enumerate(test_dataloader):
                    test_iter_loss=nnet.validate(ls1,ls_msk)
                    test_loss=test_loss+test_iter_loss
            
            print('epoch train loss = %f, epoch test loss = %f'
                %(loss_total/len(test_dataloader), test_loss/len(test_dataloader)))

            
            if display and iteration % display == 0:
                print("training loss at iteration {}: {}".format(iteration, loss_total.item()))
                
            test_loss_iter=test_loss/len(test_dataloader)
            del loss_total
            del test_loss
            
            if test_loss_iter < 0.0009:
                nnet.save_params(iteration)
                
            
            if iteration % snapshot == 0:
                nnet.save_params(iteration)
                #test()

            if iteration % stepsize == 0:
                learning_rate /= decay_rate
                nnet.set_lr(learning_rate)
                
            
                
 
 
if __name__ == "__main__":
    cfg_file = "/home/boyu/matrixnet-master/config/matrixsegnet.json"
    with open(cfg_file, "r") as f:
        configs = json.load(f)
    system_configs.update_config(configs["system"])
    train()
