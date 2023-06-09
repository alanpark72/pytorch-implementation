import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.name = "dice"

    def forward(self, input, target, smooth=1e-5):
        #if input.dim()>2:
        #    input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
        #    input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
        #    input = input.contiguous().view(-1,input.size(2))

        target_1_hot = torch.eye(2)[target.squeeze(1)]
        target_1_hot = target_1_hot.permute(0, 3, 1, 2).float()
        ##input = F.softmax(input, dim=1)
        input = torch.sigmoid(input)
        
        target_1_hot = target_1_hot.type(input.type())
        dims = (0,) + tuple(range(2, target.ndimension()))
        intersection = torch.sum(input * target_1_hot, dims)
        cardinality = torch.sum(input + target_1_hot, dims)
        dice = (2. * intersection / (cardinality + smooth)).mean() 
        
        return 1 - dice 