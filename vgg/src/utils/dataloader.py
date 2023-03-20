import torch
import numpy as np
from PIL import Image

class Dataset(torch.utils.data.Dataset):
    def __init__(self, _images, _labels, transforms):
        self.images = np.array(_images)
        self.labels = np.array(_labels)
        self.transform = transforms
        
    def __getitem__(self, idx):
        img = np.asarray(Image.open(self.images[idx]).convert('L'))
        lbl = int(np.asarray(Image.open(self.labels[idx])))
        name = str(self.images[idx])
        
        batch = {"image":img, "label":lbl, "name":name}
        
        if self.transform:
            batch = self.transform(batch)
        
        return batch
    
    def __len__(self):
        return len(self.images)