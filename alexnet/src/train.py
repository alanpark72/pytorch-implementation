import os
import numpy as np
import matplotlib.pyplot as plt

from mmcv import ProgressBar
from glob import glob
from natsort import natsorted

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, Adamax, SGD
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder

from utils.dataloader import Dataset
from utils.utils import getConfig, checkDir
from model.alexnet import AlexNet


def caseSetter(model, device, case, lr):
    if case.split("_")[0] == "Adam":
        optimizer_transfer = Adam(model.parameters(), lr=lr)
    elif case.split("_")[0] == "Adamax":
        optimizer_transfer = Adamax(model.parameters(), lr=lr)
    elif case.split("_")[0] == "SGD":
        optimizer_transfer = SGD(model.parameters(), lr=lr, weight_decay=0.0005, momentum=0.9)
    else:
        print("ERROR : Selected Optimizer is not exist.")
        exit()
    
    if case.split("_")[-1] == "XEtrp":
        criterion_transfer = CrossEntropyLoss().to(device)
    elif case.split("_")[-1] == "BCE":
        criterion_transfer = BCEWithLogitsLoss().to(device)
    else:
        print("ERROR : Selected Criterion is not exist.")
        exit()

    return optimizer_transfer, criterion_transfer

def loadBatch(config, is_classification=False):
    _transform = transforms.Compose([
            transforms.Resize((config["imgsz"],config["imgsz"])),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    
    if is_classification:
        _batch_train = ImageFolder(config["dataset"]+"train/", _transform)
        _batch_valid = ImageFolder(config["dataset"]+"valid/", _transform)
    else: # Use this option for segmentation 
        _train_imgs = natsorted(glob(config["dataset"]+"train/images/*.jpg"))
        _train_lbls = natsorted(glob(config["dataset"]+"train/labels/*.png"))
        _valid_imgs = natsorted(glob(config["dataset"]+"valid/images/*.jpg"))
        _valid_lbls = natsorted(glob(config["dataset"]+"valid/labels/*.png"))

        _batch_train = Dataset(_train_imgs, _train_lbls, _transform)
        _batch_valid = Dataset(_valid_imgs, _valid_lbls, _transform)

    train_batches = DataLoader(_batch_train, batch_size=config["batch"], shuffle=True)
    valid_batches = DataLoader(_batch_valid, batch_size=1, shuffle=False)
    
    return train_batches, valid_batches

def loadModel(weight):
    model = AlexNet(num_cls=config["num_cls"])
    
    if weight:
        if os.path.isfile(weight):
            model.load_state_dict(torch.load(weight))
            print("Weight file {} load successfully.".format(weight))
        else:
            print("Weight file {} is missing.\nPlease check the file name.".format(weight))
    else :
        print("Disable using pretrained weight.")
        
    return model

def train(model, train_batches, valid_batches, epochs, opt, cri, device, use_cuda, save_path):
    if use_cuda:
        model = model.cuda()
    
    min_val_loss = np.Inf
    early_stop = config["early_stop"]
    cnt = 0
    
    lst_tr_loss = []
    lst_val_loss = []
    
    print("\nTraining Start.\n")
    
    for epoch in range(epochs):
        tr_loss = 0.
        val_loss = 0.
        
        weight_path = save_path+"weights/"
        
        print("Epoch #{}".format(epoch))
        print("Train Batches")
        
        model.train()
        tr_bar = ProgressBar(len(train_batches))
        
        for data in train_batches:
            image = data[0].to(device, dtype=torch.float32)
            target = data[1].to(device, dtype=torch.long)
            
            opt.zero_grad()
            
            output = model(image)
            
            loss = cri(output, target)
            loss.backward()
            
            opt.step()
            
            tr_loss += loss.item()*image.size(0)
            tr_bar.update()
            
            torch.cuda.empty_cache()
        
        print("\nValidation Batches")
        
        model.eval()
        val_bar = ProgressBar(len(valid_batches))
        
        for data in valid_batches:
            image = data[0].to(device, dtype=torch.float32)
            target = data[1].to(device, dtype=torch.long)
            
            output = model(image)
            loss = cri(output, target)
            
            val_loss += loss.item()*image.size(0)
            val_bar.update()
        print("") # ProgressBar
        
        tr_loss = tr_loss/len(train_batches.sampler)
        val_loss = val_loss/len(valid_batches.sampler)
        lst_tr_loss.append(tr_loss)
        lst_val_loss.append(val_loss)
        
        if val_loss < min_val_loss:
            checkDir(weight_path)
            weight_path += "epoch{}.pt".format(epoch)
            torch.save(model.to(torch.float16).state_dict(), weight_path)
            model.to(torch.float32)
            print("Validation loss decreased ({:.10f} --> {:.10f})\nSaving current model file at: {}".format(min_val_loss, val_loss, weight_path))
            min_val_loss = val_loss
            cnt = 0
        elif epoch == epochs-1:
            checkDir(weight_path)
            weight_path += "last.pt"
            torch.save(model.state_dict(), weight_path)
            print("Training End.")
        else:
            cnt += 1
            if cnt >= early_stop:
                weight_path += "last.pt"
                #torch.save(model.state_dict(), weight_path)
                torch.save(model.to(torch.float16).state_dict(), weight_path)
                model.to(torch.float32)
            
                print("Result #{}\tTraining Loss: {:.10f}\tValidation Loss: {:.10f}\tLearning Rate: {}\n".format(epoch,tr_loss,val_loss,config["lr"]))
                print("Early stop training at epoch: {}".format(epoch))
                break
            
        print("Result #{}\tTraining Loss: {:.10f}\tValidation Loss: {:.10f}\tLearning Rate: {}\n".format(epoch,tr_loss,val_loss,config["lr"]))
    
    x_len = np.arange(len(lst_tr_loss))
    
    plt.plot(x_len, lst_tr_loss, marker='.', c='blue', label="Train Loss")
    plt.plot(x_len, lst_val_loss, marker='.', c='orange', label="Validation Loss")
    plt.legend(loc='upper right')
    plt.grid()
    plt.title("{}-{}".format(config["name"], config["case"]))
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(save_path+"{}-{}.png".format(config["name"], config["case"]))


def run(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_cuda = torch.cuda.is_available()
    print("On Device : ", device)
    
    model = loadModel(config["weight"])
    train_batches, valid_batches = loadBatch(config, is_classification=True)
    opt, cri = caseSetter(model, device, config["case"], config["lr"])
    
    result_path = "./result/{}/".format(config["name"])
    result_path = checkDir(result_path, auto_increment=True)
    save_path = result_path + "train/"
    checkDir(save_path)
    
    with open("{}opt.txt".format(save_path), "w") as f:
        f.writelines("dataset: {}\nname: {}\nweight: {}\nnum_cls: {}\nearly_stop: {}\nimgsz: {}\nbatch: {}\nepoch: {}\nlr: {}\ncase: {}\n".format(
                    config["dataset"], config["name"], config["weight"], config["num_cls"], config["early_stop"], config["imgsz"], config["batch"], config["epoch"], config["lr"], config["case"]))
        f.close()
    
    train(model, train_batches, valid_batches, config["epoch"], opt, cri, device, use_cuda, save_path)


if __name__=="__main__":
    config = getConfig(path="./config/sample_train.yaml")
    run(config)