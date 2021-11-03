#refectoring 하기 
#config -> main으로 올려서 

import argparse
import torch

#Training
from tasks.whole_train import wholetrain


def main(argsm, device):

    #train
    wholetrain(args)


if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    
    #set tensorboardX path 
    parser.add_argument('--record_dir', type=str)

    #dataset path
    parser.add_argument('--anime_dataset_path', type=str)
    parser.add_argument('--inter_frames', type=int)
    
    #RFR module setting 
    parser.add_argument('--whole_RFR_epoch', type=int, default=150)

    #whole train setting
    parser.add_argument('--whole_train_epoch', type=int, default=50)
    parser.add_argument('--whole_train_checkpoint', type=str)
    parser.add_argument('--whole_train_checkpoint_epoch', type=int)
    parser.add_argument('--whole_train_model', type=str, default='AnimeInterp_no_cupy')
    parser.add_argument('--whole_train_init_learning_rate', type=float)
    parser.add_argument('--whole_train_checkpoint', type=float)
    parser.add_argument('--whole_train_resume', action='store_true')
    parser.add_argument('--whole_train_width', type=int, default=640)
    parser.add_argument('--whole_train_height', type=int, default=360)
    parser.add_argument('--whole_train_crop', type=int, default=352)
    args = parser.parse_args()

    main(args, device)

