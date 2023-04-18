import os

from mmcv import ProgressBar
from glob import glob
from natsort import natsorted

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder

from utils.dataloader import Dataset
from utils.utils import getConfig
from model.vgg import VGG19, VGG16


def loadBatch(config, is_classification=False):
    _transform = transforms.Compose([
            transforms.Resize((config["imgsz"],config["imgsz"])),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    
    if is_classification:
        _batch_test = ImageFolder(config["dataset"]+"test/", _transform)
    else: # Use this option for segmentation 
        _test_imgs = natsorted(glob(config["dataset"]+"test/images/*.jpg"))
        _test_lbls = natsorted(glob(config["dataset"]+"test/labels/*.png"))
        _batch_test = Dataset(_test_imgs, _test_lbls, _transform)

    test_batches = DataLoader(_batch_test, batch_size=1, shuffle=False)
    
    return test_batches, _batch_test.classes

def loadModel(weight, type="vgg19"):
    if type == "vgg19":
        model = VGG19(num_cls=config["num_cls"])
    else:
        model = VGG16(num_cls=config["num_cls"])
    
    if weight:
        if os.path.isfile(weight):
            model.load_state_dict(torch.load(weight))
            print("Weight file {} load successfully.".format(weight))
        else:
            print("Weight file {} is missing.\nPlease check the file name.".format(weight))
    else :
        print("Disable using pretrained weight.")
        
    return model

def classify(model, test_batches, cls_lbl, device, use_cuda):
    if use_cuda:
        model = model.cuda()
        model.eval()
    
    correct_cnt = 0
    
    print("\nClassification")
    
    with torch.no_grad():
        test_bar = ProgressBar(len(test_batches))
        
        for data in test_batches:
            image = data[0].to(device, dtype=torch.float32)
            target = data[1].to(device, dtype=torch.long)
            
            output = model(image)
            probs = torch.softmax(output, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            if target.item() == preds.item():
                correct_cnt += 1
            
            #print(cls_lbl[target.item()], cls_lbl[preds.item()])
            
            test_bar.update()
        print("") # ProgressBar
    
    print("Model Accuracy: {:.3f}%".format(correct_cnt/len(test_batches)*100))
    
def run(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_cuda = torch.cuda.is_available()
    print("On Device : ", device)
    
    model = loadModel(config["weight"], type=config["model"]) ## type: 19 or 16 (VGG-19 or VGG-16)
    test_batches, cls_lbl = loadBatch(config, is_classification=True)
    
    classify(model, test_batches, cls_lbl, device, use_cuda)
    
if __name__=="__main__":
    config = getConfig(path="./config/sample_classify.yaml")
    run(config)