import os

from natsort import natsorted
from glob import glob

import torch
from torch.utils.data import random_split

from utils import checkDir


def splitDataset(data, ratio=[0.8,0.1,0.1]):
    len_train = int(len(data)*ratio[0])
    len_test = int(len(data)*ratio[1])
    len_valid = len(data) - (len_train+len_test) # Not be using assigned ratio because of its rounding issue.
    
    train, test, valid = random_split(data, [len_train, len_test, len_valid], torch.Generator().manual_seed(42))
    
    train_path = data_path + "train/"
    checkDir(train_path)

    test_path = data_path + "test/"
    checkDir(test_path)

    valid_path = data_path + "valid/"
    checkDir(valid_path)
    
    for _src in train:
        _path = _src.split("/")
        _lbl = _path[2]
        _file = _path[-1]
        _train_path = train_path + _lbl + "/"
        checkDir(_train_path)
        n_path = _train_path + _file
        
        os.rename(_src, n_path)
    
    for _src in test:
        _path = _src.split("/")
        _lbl = _path[2]
        _file = _path[-1]
        _test_path = test_path + _lbl + "/"
        checkDir(_test_path)
        n_path = _test_path + _file
        
        os.rename(_src, n_path)
    
    for _src in valid:
        _path = _src.split("/")
        _lbl = _path[2]
        _file = _path[-1]
        _valid_path = valid_path + _lbl + "/"
        checkDir(_valid_path)
        n_path = _valid_path + _file
        
        os.rename(_src, n_path)


if __name__ == "__main__":
    data_path = "./dataset/"
    data = natsorted(glob(data_path+"*/*.jpg")) + natsorted(glob(data_path+"*/*.png"))
    
    splitDataset(data, ratio=[0.8,0.1,0.1])