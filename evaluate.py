import sys

from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import datas.datasets as datasets
from utils import flow_viz
from utils import frame_utils
from tasks.evaluate import validate_chairs, validate_kitti, validate_sintel 

from utils.utils import InputPadder, forward_interpolate
from models.rfr_model.rfr_new import RFR 





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint", default=r'C:\Users\user\Desktop\graduation\checkpoints\rfr_sintel_latest.pth')
    parser.add_argument('--dataset', help="dataset for evaluation", default='sintel')
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    model = RFR(args)
    #model = nn.DataParallel(model)
    checkpoint = torch.load(args.model)
    #model.load_state_dict(checkpoint['model'])
    model.load_state_dict(checkpoint, strict=False)

    model.cuda()
    model.eval()

    #create_sintel_submission(model, warm_start=True)
    # create_kitti_submission(model.module)

    with torch.no_grad():
        if args.dataset == 'chairs':
            validate_chairs(model)

        elif args.dataset == 'sintel':
            validate_sintel(model)

        elif args.dataset == 'kitti':
            validate_kitti(model)
