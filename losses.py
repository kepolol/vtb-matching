import torch.nn as nn
import torch.nn.functional as f


class ContrastiveLoss(nn.Module):

    # https://github.com/adambielski/siamese-triplet/blob/master/losses.py
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, raw=False):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * f.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        if raw:
            return losses
        else:
            return losses.mean()
