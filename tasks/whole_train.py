import models
import datas 
import argparse
import torch
import torchvision.transforms as TF
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F 
import os
from math import log10 
import numpy as np
from utils.config import Config
import sys
import cv2 as cv 
from utils.vis_flow import flow_to_color 
import json
from skimage.measure import compare_psnr, compare_ssim 
from configs import train_config

def whole_train(args):
    #preparing datasets & normalization
    normalize1 = TF.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    normalize2 = TF.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    trans = TF.Compose([TF.ToTensor(), normalize1, normalize2, ])

    #용도 알면 추후에 적어두기
    revmean = [-x for x in [0.0, 0.0, 0.0]]
    revstd = [1.0/ x for x in [1.0, 1.0, 1.0]]
    revnormalize1 = TF.Normalize([0.0,0.0,0.0], revstd)
    revnormalize2 = TF.Normalize(revmean, [1.0, 1.0, 1.0])
    revNormalize = TF.Compose([revnormalize1, revnormalize2])

    revtrans = TF.Compose([revnormalize1, revnormalize2, TF.ToPILImage()])

    while_trainset = datas.AniTriplet(root = train_config.trainset_root, transform = trans, train=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for epoch in range(train_)

    

