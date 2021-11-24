from __future__ import print_function, division
import sys
import argparse
import os
import cv2 as cv
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.rfr_model.rfr_new import RFR 
from models.rfr_nc_model.rfr_new_nc import RFR_nc
from models.rfr_model.rfr_new_gma import RFR_gma

import tasks.evaluate as evaluate 

import datas.datasets as datasets




try:

    from torch.cuda.amp import GradScaler

except:

    # dummy GradScaler for PyTorch < 1.6

    class GradScaler:
        def __init__(self):
            pass

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            pass

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            pass

# exclude extremly large displacements

MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000





from torch.utils.tensorboard import SummaryWriter

def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)    
    flow_loss = 0.0
    # exlude invalid pixels and extremely large diplacements
    
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)
    

    for i in range(n_predictions):

        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()
    
    #need to check 
    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]
    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics

def show_image(img):
    img = img.permute(1,2,0).cpu().numpy()
    plt.imshow(img/255.0)
    plt.show()
    # cv2.imshow('image', img/255.0)
    # cv2.waitKey()



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    if args.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.whole_RFR_lr, weight_decay=args.wdecay, eps = args.epsilon)
    elif args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)
    else:
        raise NotImplementedError('{} optimizer is not implemented!'.format(args.optimizer))

    if args.scheduler.lower() == 'cyclic':
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.whole_RFR_lr, args.num_steps+100,
                                                  pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
    elif args.scheduler.lower() == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=0.5)
    else:
        raise NotImplementedError('{} scheduler is not implemented!'.format(args.scheduler))
    return optimizer, scheduler

def plot_val(logger, args):
    for key in logger.val_results_dict.keys():
        # plot validation curve
        plt.figure()
        plt.plot(logger.val_steps_list, logger.val_results_dict[key])
        plt.xlabel('x_steps')
        plt.ylabel(key)
        plt.title(f'Results for {key} for the validation set')
        plt.savefig(args.rfr_checkpoint_save_path+f"/{key}.png", bbox_inches='tight')
        plt.close()


def plot_train(logger, args):
    # plot training curve
    plt.figure()
    plt.plot(logger.train_steps_list, logger.train_epe_list)
    plt.xlabel('x_steps')
    plt.ylabel('EPE')
    plt.title('Running training error (EPE)')
    plt.savefig(args.rfr_checkpoint_save_path+"/train_epe.png", bbox_inches='tight')
    plt.close()





class Logger:

    def __init__(self, model, scheduler):

        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None



    def _print_training_status(self):

        metrics_data = [self.running_loss[k]/SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[total_steps: {:6d}, last_lr: {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)

        
        # print the training status

        print(training_str + metrics_str)

        if self.writer is None:

            self.writer = SummaryWriter()



        for k in self.running_loss:

            self.writer.add_scalar(k, self.running_loss[k]/SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0



    def push(self, total_steps, metrics):

        self.total_steps = total_steps

        for key in metrics:

            if key not in self.running_loss:
                self.running_loss[key] = 0.0

        self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):

        if self.writer is None:
            self.writer = SummaryWriter()



        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)



    def close(self):
        self.writer.close()





def rfr_train(args, device):

    if args.model=='rfr_new' : 
        model = RFR(args)
    elif args.model== 'rfr_new_nc':
        model = RFR_nc(args)
    elif args.model == 'rfr_new_gma':
        model = RFR_gma(args)
    
    print("Parameter Count: %d" % count_parameters(model))

    total_steps = 0
    optimizer, scheduler = fetch_optimizer(args, model)
    scaler = GradScaler(enabled=args.mixed_precision) # 이 부분 찾아보기 

    model = model.to(device)
    model.train()
    print(args.whole_RFR_checkpoint)

    if args.whole_RFR_checkpoint is not None:
        checkpoint = torch.load(args.whole_RFR_checkpoint)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        total_steps = checkpoint['total_steps'] + 1
        print(checkpoint['total_steps'] )

    if args.stage != 'chairs':
        model.freeze_bn()

    train_loader = datasets.fetch_dataloader(args)
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler)

    VAL_FREQ = 5000
    
    should_keep_training = True
    
    while should_keep_training:

        for i_batch, data_blob in enumerate(train_loader):

            optimizer.zero_grad()
            image1, image2, flow, valid = [x.to(device) for x in data_blob]

            if args.add_noise:

                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)
            
            #if args.training and args.requires_sq_flow:
                #flow_predictions = model(image1, image2,iters=args.iters)            
            #else:
            flow_predictions = model(image1, image2,iters=args.iters)     

            loss, metrics = sequence_loss(flow_predictions, flow, valid, args.gamma)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)                

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            
            logger.push(total_steps,metrics)

            if total_steps % VAL_FREQ == 0:

                PATH = os.path.join(args.rfr_checkpoint_save_path,'{}_{}.pth'.format(total_steps+1, args.name))
                results = {}
                for val_dataset in args.validation:

                    if val_dataset == 'chairs':
                        results.update(evaluate.validate_chairs(args, model))

                    elif val_dataset == 'sintel':
                        results_t = evaluate.validate_sintel(model)
                        results.update(evaluate.validate_sintel(model))

                    elif val_dataset == 'kitti':
                        results.update(evaluate.validate_kitti(model))

                logger.write_dict(results)

                torch.save(
                        {'total_steps':total_steps,
                         'model':model.state_dict(),
                         'optimizer':optimizer.state_dict(),
                         'scheduler':scheduler.state_dict(),
                         'scaler':scaler.state_dict()
                        }, PATH)

                model.train()

                if args.stage != 'chairs':
                    model.freeze_bn()

            total_steps += 1

            if logger.total_steps % args.val_freq == args.val_freq - 1:
                evaluate(model, args, logger)
                plot_train(logger, args)
                plot_val(logger, args)
                PATH = os.path.join(args.whole_train_checkpoint, '/{logger.total_steps+1}_{args.name}.pth')
                torch.save(model.state_dict(), PATH)

            if total_steps > args.num_steps:
                
                should_keep_training = False
                PATH = os.path.join(args.rfr_checkpoint_save_path,'{}_{}.pth'.format('final',args.name))
                torch.save(
                        {'total_steps':total_steps,
                        'model':model.state_dict(),
                        'optimizer':optimizer.state_dict(),
                        'scheduler':scheduler.state_dict(),
                        'scaler':scaler.state_dict()}, PATH)

                print('end training!')

                break

            
    logger.close()

    return PATH








