import os

import cv2
import yaml
import json
import numpy as np

from glob import glob
from natsort import natsorted
from PIL import Image
from mmcv import ProgressBar

import torch
from torch.utils.data import random_split

from utils import checkDir, getConfig


"""
============== Function List ==============
0. Renamer
1. Split Dataset
2. Data Shift
3. Label mask Generator for Segmentation
4. Ground Truth Generator
5. Label mask Generator for Segmentation (with Resize)
6. Generate Class Weight
===========================================
"""


func = 1

data_path = "./dataset/"

lst_og = natsorted(glob(data_path+"*/*.jpg")) + natsorted(glob(data_path+"*/*.png"))

"""base_path = "./dataset/rs/"

train_bs_path = base_path + "train/"
tr_im_path = train_bs_path + "images/"
checkDir(tr_im_path)
tr_label_path = train_bs_path + "labels/"
checkDir(tr_label_path)
tr_gt_path = train_bs_path + "gt/"
checkDir(tr_gt_path)

test_bs_path = base_path + "test/"
te_im_path = test_bs_path + "images/"
checkDir(te_im_path)
te_label_path = test_bs_path + "labels/"
checkDir(te_label_path)
te_gt_path = test_bs_path + "gt/"
checkDir(te_gt_path)

valid_bs_path = base_path + "valid/"
val_im_path = valid_bs_path + "images/"
checkDir(val_im_path)
val_label_path = valid_bs_path + "labels/"
checkDir(val_label_path)
val_gt_path = valid_bs_path + "gt/"
checkDir(val_gt_path)

im_size = (1080,1920,3)

tr_im_glob = natsorted(glob(train_path+"*.png"))
tr_js_glob = natsorted(glob(train_path+"*.json"))
te_im_glob = natsorted(glob(test_path+"*.png"))
te_js_glob = natsorted(glob(test_path+"*.json"))
val_im_glob = natsorted(glob(valid_path+"*.png"))
val_js_glob = natsorted(glob(valid_path+"*.json"))"""

COLOR_CLS = {"instrument":(255,0,0), "bladder":(0,0,255), "prostate":(0,255,0),"vas":(255,0,255), "sv":(255,0,255), "suction":(255,0,0)} ## sv -> vas
COLOR = {0:(0,0,0), 1:(0,0,255), 2:(255,0,0), 3:(0,255,0), 4:(255,0,255)}


def renamer(lst_og):
    for _file in lst_og:
        n_file = _file
        json_file = _file.replace(os.path.splitext(_file)[-1], ".json")
        
        if ' ' in n_file:
            n_file = _file.replace(' ', '')
            
        n_name,n_ext = os.path.splitext(os.path.basename(n_file))
        
        tag = os.path.dirname(_file).split("/")[-1]
        
        n_file = n_file.replace(n_name+n_ext, n_name+"_"+tag+n_ext)
        n_json_file = n_file.replace(n_ext, ".json")
        
        print("rename "+_file+" to "+n_file)
        
        os.rename(_file, n_file)
        os.rename(json_file, n_json_file)

def splitDataset(lst_og, ratio=[0.8,0.1,0.1]):
    len_train = int(len(lst_og)*ratio[0])
    len_test = int(len(lst_og)*ratio[1])
    len_valid = len(lst_og) - (len_train+len_test)
    
    train, test, valid = random_split(lst_og, [len_train, len_test, len_valid], torch.Generator().manual_seed(42))
    
    train_path = data_path + "train/"
    checkDir(train_path)

    test_path = data_path + "test/"
    checkDir(test_path)

    valid_path = data_path + "valid/"
    checkDir(valid_path)
    
    for _file in train:
        _path = _file.split("/")
        _train_path = train_path + _path[2] + "/"
        checkDir(_train_path)
        n_path = _train_path + _path[-1]
        
        os.rename(_file, n_path)
    
    for _file in test:
        _path = _file.split("/")
        _test_path = test_path + _path[2] + "/"
        checkDir(_test_path)
        n_path = _test_path + _path[-1]
        
        os.rename(_file, n_path)
    
    for _file in valid:
        _path = _file.split("/")
        _valid_path = valid_path + _path[2] + "/"
        checkDir(_valid_path)
        n_path = _valid_path + _path[-1]
        
        os.rename(_file, n_path)

def genLabelMask(lst_js, lbl_path):
    for i, js in enumerate(lst_js):
        fname = js.split('/')[-1][:-5]
        mask = np.zeros(im_size, dtype=np.uint8)
        with open(js) as js:
            js = json.load(js)
            pols = js["shapes"]

            for pol in pols:
                lb = pol["label"]
                pts = np.array(pol["points"], dtype=np.int32)
                
                #mask = cv2.fillConvexPoly(mask, pts, (255,0,0))
                if lb == "Instrument":
                    lb = "instrument"
                mask = cv2.fillPoly(mask, [pts], COLOR_CLS[lb])

            cv2.imwrite(lbl_path+fname+".png", mask)

def getPoints(pts):
    x_min = pts[pts.argmin(axis=0)[0]][0]
    x_max = pts[pts.argmax(axis=0)[0]][0]
    y_min = pts[pts.argmin(axis=0)[1]][1]
    y_max = pts[pts.argmax(axis=0)[1]][1]

    x_ct = int((x_min+x_max)/2)
    y_ct = int((y_min+y_max)/2)

    ct = (x_ct, y_ct)

    return ct

def putLegend(src, polygons, color):
    _base = np.full((300,270,3),(255,255,255),dtype=np.uint8)
    offset = 10
    lb = set([p["label"] for p in polygons]) # set -> 중복제거

    space = int(300/(len(lb)*2))

    for i, l in enumerate(lb):
        if l == "Instrument":
            l = "instrument"
        _base = cv2.rectangle(_base, (10,int(space/2)+(space*2*i)), (100,int(space/2)+(space*(2*i+1))), color[l], -1 )
        _base = cv2.putText(_base, l, (110,int(space/2)+(space*(2*i+1))-offset), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,0,0), 2)
    
    src[offset:300+offset,1920-270-offset:1920-offset] = _base

def genGroundTruth(lst_js, gt_path):
    for i, js in enumerate(lst_js):
        src = cv2.imread(tr_im_glob[i])
        fname = js.split('/')[-1][:-5]

        with open(js) as js:
            js = json.load(js)
            pols = js["shapes"]
            
            putLegend(src,pols,COLOR_CLS)

            for pol in pols:
                lb = pol["label"]
                pts = np.array(pol["points"], dtype=np.int32)
                
                if lb == "Instrument":
                    lb = "instrument"    

                ct = getPoints(pts)

                #mask = cv2.fillConvexPoly(mask, pts, (255,0,0))
                rst = cv2.polylines(src, [pts], True, COLOR_CLS[lb], 2)

            cv2.imwrite(gt_path+fname+".png", rst)

def resizeImage(src, size=(1920,1080)):
    mask = np.zeros(size, dtype=np.uint8)
    w,h,c = src.shape
    test=1

def genResizedLabelMask(lst_js, lbl_path):
    for i, js in enumerate(lst_js):
        fname = js.split('/')[-1][:-5]
        
        if os.path.isfile(lbl_path+fname+".png"):
            continue
        
        mask = np.zeros(im_size, dtype=np.uint8)
        img = js.replace(".json",".png")
        img = cv2.imread(img)
        nh,nw = 0,0
        
        if img.shape[0] == 1024:
            for i in range(1, im_size[0] + 1):
                if (im_size[0] % i == 0) & (img.shape[0] % i == 0):
                    gcd = i
                    
            nh,nw = im_size[0], img.shape[1]//(img.shape[0]//gcd)*(im_size[0]//gcd)
            mask = np.zeros_like(img, dtype=np.uint8)
            img = cv2.resize(img, (nw,nh), interpolation=cv2.INTER_LINEAR)
            sy,sx = int((im_size[0]-nh)//2), int((im_size[1]-nw)//2)
            ey,ex = sy+nh, sx+nw
            
            o_mask = np.zeros(im_size, dtype=np.uint8)
            o_img = o_mask.copy()
        
        with open(js) as jsf:
            jsf = json.load(jsf)
            pols = jsf["shapes"]

            for pol in pols:
                lb = pol["label"]
                pts = np.array(pol["points"], dtype=np.int32)
                
                #mask = cv2.fillConvexPoly(mask, pts, (255,0,0))
                if lb == "Instrument":
                    lb = "instrument"
                mask = cv2.fillPoly(mask, [pts], COLOR_CLS[lb])
        
        if not nh == 0:
            mask = cv2.resize(mask, (nw,nh), interpolation=cv2.INTER_LINEAR)
            o_mask[sy:ey, sx:ex] = mask
            cv2.imwrite(lbl_path+fname+".png", o_mask)
            o_img[sy:ey, sx:ex] = img
            cv2.imwrite((lbl_path+fname+".png").replace("labels", "images"), o_img)
        else:
            cv2.imwrite(lbl_path+fname+".png", mask)

def rgb2mask(src):
    mask = np.zeros((src.shape[0], src.shape[1]))
    for cls, color in COLOR.items():
        mask[np.all(src==color, axis=2)] = cls
    
    return mask

def genClassWeight(config):
    #spc = np.zeros(config["num_cls"], dtype=np.uint8) # samples per class
    spc = torch.zeros(config["num_cls"])
    _lbl_path = config["dataset"] + "train/labels/"
    lst_lbl = natsorted(glob(_lbl_path+"*.png"))
    
    _bar = ProgressBar(len(lst_lbl))
    
    for _lbl in lst_lbl:
        _lbl = np.asarray(Image.open(_lbl).convert('RGB'))
        lbl = rgb2mask(_lbl)
        cls_cnt = np.unique(lbl).astype(np.uint8)
        
        for _idx in cls_cnt:
            spc[int(_idx)] += 1
            
        _bar.update()
    
    cls_weight = len(lst_lbl) / (config["num_cls"] * spc)
    config["cnt_cls"] = list(map(lambda x: int(x), spc))
    config["cls_weight"] = list(map(lambda x: round(float(x), 4), cls_weight))
    
    with open('./config/config.yaml', 'w') as file:
        yaml.dump(config, file, default_flow_style=None)


if __name__ == "__main__":
    config = getConfig()
    
    if func == 0:
        renamer(lst_og)
    elif func == 1:
        splitDataset(lst_og, [0.8,0.1,0.1])
    elif func == 2:
        os.system("cp -r {}* {}".format(train_path, tr_im_path))
        os.system("cp -r {}* {}".format(test_path, te_im_path))
        os.system("cp -r {}* {}".format(valid_path, val_im_path))
    elif func == 3:
        genLabelMask(tr_js_glob, tr_label_path)
        genLabelMask(te_js_glob, te_label_path)
        genLabelMask(val_js_glob, val_label_path)
    elif func == 4:
        genGroundTruth(tr_js_glob, tr_gt_path)
        genGroundTruth(te_js_glob, te_gt_path)
        genGroundTruth(val_js_glob, val_gt_path)
    elif func == 5:
        genResizedLabelMask(tr_js_glob, tr_label_path)
        genResizedLabelMask(te_js_glob, te_label_path)
        genResizedLabelMask(val_js_glob, val_label_path)
    elif func == 6:
        genClassWeight(config)
    