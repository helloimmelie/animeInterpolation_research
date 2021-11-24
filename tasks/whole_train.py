import models
import datas 
import argparse
import torch
import torchvision.transforms as TF
import torchvision 
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
import time
import datetime
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tensorboardX import SummaryWriter
from test_anime_sequence_one_by_one import validate



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

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def validate(args, device, model):
    retImg = []
    # For details see training.
    psnr = 0
    psnrs = [0 , 0]
    ssims = [0, 0]
    tloss = 0
    tlosses = [0, 0]
    flag = True
    retImg = []

    testset = datas.AniTripletWithSGMFlowTest(args.testset_root, args.test_flow_root, trans, args.test_size, args.test_crop_size, train=False)
    sampler = torch.utils.data.SequentialSampler(testset)
    validationloader = torch.utils.data.DataLoader(testset, sampler=sampler, batch_size=1, shuffle=False, num_workers=1)

    with torch.no_grad():

        folders = []
        
        for validationIndex, validationData in enumerate(validationloader, 0):
            sys.stdout.flush()
            sample, flow,  index, folder = validationData

            frame0 = None
            frame1 = sample[0]
            frame3 = None
            frame2 = sample[-1]
            frameT = sample[1]

            folders.append(folder[0][0])
            
            # initial SGM flow
            F12i, F21i  = flow

            F12i = F12i.float().cuda() 
            F21i = F21i.float().cuda()

            ITs = [sample[tt] for tt in range(1, 2)]
            I1 = frame1.cuda()
            I2 = frame2.cuda()
            It_warps = []
            Ms = []

            for tt in range(args.inter_frames):
                x = args.inter_frames
                t = 1.0/(x+1) * (tt + 1)
                
                outputs = model(I1, I2, F12i, F21i, t)
                It_warp = outputs
             
                It_warps.append(It_warp)

                loss = F.l1_loss(outputs, frameT)
                tlosses[tt] += loss.item()

            # record psnrs 
                psnrs[tt] += peak_signal_noise_ratio()
                ssims[tt] += structural_similarity()

	        # record interpolated frames 
            img_grid = []
            img_grid.append(revNormalize(frame1[0]))
            for tt in range(7):
                img_grid.append(revNormalize(It_warps[tt].cpu()[0]))
            img_grid.append(revNormalize(frame2[0]))

            retImg.append(torchvision.utils.make_grid(img_grid, nrow=10, padding=10))

        for tt in range(7):
            psnrs[tt] /= len(validationloader)
            tlosses[tt] /= len(validationloader)

    return psnrs, tlosses, retImg


def whole_train(args,device):

  
   

    #Model initiating
    whole_trainset = datas.AniTripletWithSGMFlow(root = args.anime_dataset_path, transform = trans, resizeSize = (args.whole_train_size_width, args.whole_train_size_height), randomCropSize = (args.whole_train_crop_width, args.whole_train_crop_width), train=True)
    trainloader = torch.utils.data.DataLoader(whole_trainset,  batch_size = args.whole_train_batch_size, shuffle=True, num_workers=0)
    train_model = getattr(models, args.whole_train_model)(args.pwc_path) #pwc_path: checkpoint path 
    model = nn.DataParallel(train_model)
    
    
    if args.whole_train_resume:
        dict1 = torch.load(args.whole_train_checkpoint)

    #optimizer
    optimizer = optim.Adam(model.params, lr=args.whole_train_init_learning_rate)

    # scheduler to decreate learning rate
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.3)
    recorder = SummaryWriter(args.record_dir)

        

    start = time.time()
    cLoss = dict1['loss']
    valLoss = dict1['valLoss']
    valPSNR = dict1['valPSNR']
    valSSIM = dict1['valSSIM']
    checkpoint_counter = 0
    

        
    
    for epoch in range(dict1['epoch'] + 1, args.whole_train_epoch):

        print("epoch: ", epoch)

        cLoss.append([])
        valLoss.append([])
        valPSNR.append([])
        valSSIM.append([])
        iLoss = 0

        folders = []

        for trainIndex, trainData in enumerate(trainloader, 0):
            
            print(trainIndex, len(trainloader))#이 부분 고민해보기(어떻게 출력할지?) 

            sample, flow = trainData
            
            frame0 = sample[0]
            frame1 = sample[-1]
            frameT = sample[1]


            # initial SGM flow
            F12i, F21i = flow

            F12i = F12i.float().to(device)
            F21i = F21i.float().to(device)
            
            
            I1 = frame0.to(device)
            I2 = frame1.to(device)
            IT = frameT.to(device)
    
            optimizer.zero_grad()
            
            #이 부분 로직을 아직 이해 못하겠다.. 
            for tt in range(args.inter_frames): #debug 걸어보기 
                x = args.inter_frames
                t = 1.0/(x+1) * (tt + 1)
                output = model(I1, I2, F12i, F21i, t)
                loss = F.l1_loss(output,IT) #loss function:l1 loss(prediction and groundtruth)
                loss.backward()
                optimizer.step()

                iLoss += loss.item()                
            
            if((trainIndex % args.progress_iter) == args.progress_iter - 1):
                end = time.time()

                psnrs, ssims, vLoss = validate(args, device, train_model)
                
                valPSNR[epoch].append(psnrs)
                valSSIM[epoch].append(ssims)
                valLoss[epoch].append(vLoss)

                #Tensorboard
                itr = trainIndex + epoch*(len(trainloader))

                recorder.add_scalars('Loss', {'trainLoss': iLoss/args.progress_iter, 'validationLoss':vLoss}, itr)

                vtdict={}
                psnrdict={}
                ssimdict={}
                for tt in range(args.inter_frames):
                    vtdict['validationLoss' + str(tt+1)] = vLoss[tt]
                    psnrdict['PSNR'+str(tt+1)] = psnrs[tt]
                    ssimdict['SSIM'+str(tt+1)] = ssims[tt]
                
                recorder.add_scalars('Losst', vtdict, itr)
                recorder.add_scalars('PSNRt', psnrdict, itr)
                recorder.add_scalars('SSIMt', ssimdict, itr)

                endVal = time.time()

                print(" Loss: %0.6f  Iterations: %4d/%4d  TrainExecTime: %0.1f  ValLoss:%0.6f  ValPSNR: %0.4f  ValEvalTime: %0.2f LearningRate: %f" % (iLoss / args.progress_iter, trainIndex, len(trainloader), end - start, vLoss, args.psnr, endVal - end, get_lr(optimizer)))
                sys.stdout.flush()

                cLoss[epoch].append(iLoss/args.progress_iter)
                iLoss = 0
                start = time.time()


        #Append and reset
        

        #Create checkpoint after every 'args.checkpoint_epoch' epochs 
        if ((epoch % args.checkpoint_epoch) == args.checkpoint_epoch - 1):
            dict1 = {
                'Detail':"whole_train",
                'epoch':epoch,
                'timestamp':datetime.datetime.now(),
                'trainBatchSz':args.train_batch_size,
                'validationBatchSz':1,
                'learningRate':get_lr(optimizer),
                'loss':cLoss,
                'valLoss':valLoss,
                'valPSNR':valPSNR,
                'valSSIM':valSSIM, 
                'model_state_dict' : model.state_dict(),
                    }
            torch.save(dict1, os.path.join(args.whole_train_checkpoint_dir), 'whole_model', str(checkpoint_counter)+".tar.pth")
            checkpoint_counter += 1

        #Increment scheduler count
        scheduler.step()
        
