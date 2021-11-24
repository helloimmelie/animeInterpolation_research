#refectoring 하기 
#config -> main으로 올려서 

import argparse
import torch
import sys

#Training
from tasks.whole_train import whole_train
from tasks.rfr_train import rfr_train

def main(argsm, device):

    #train
    rfr_train(args, device)
    #whole_train(args)


if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()

    #exp_name
    parser.add_argument('--name', type=str)

    #optimizer and scheduler
    parser.add_argument('--optimizer', default='adamw')
    parser.add_argument('--scheduler', default='cyclic')
    parser.add_argument('--gamma', type=float,  default=0.8, help='exponential weighting')

    #set tensorboardX path 
    parser.add_argument('--record_dir', type=str)

    #dataset path
    parser.add_argument('--anime_dataset_path', type=str)
    parser.add_argument('--inter_frames', type=int)
    
    #RFR module setting 
    parser.add_argument('--num_steps', type=int, default=120000)
    parser.add_argument('--stage', default='chairs', help='determines which dataset to use for training')
    parser.add_argument('--validation', type=str, default=['sintel'], nargs='+')#?
    parser.add_argument('--rfr_dataset_path',default=r'C:\Users\user\Desktop\graduation\datasets\FlyingChairs\FlyingChairs_release', type=str)
    parser.add_argument('--whole_RFR_lr', type=float, default=0.00025)
    parser.add_argument('--whole_rfr_batch_size', type=int, default=1)
    parser.add_argument('--image_size', type=int, nargs='+', default=[368, 496])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--training', action='store_true')

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--val_freq', type=int, default=10000, help='validation frequency')
    parser.add_argument('--print_freq', type=int, default=100, help='printing frequency')
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--add_noise', action='store_true')
    parser.add_argument('--flow_init',action='store_true')

    parser.add_argument('--whole_RFR_checkpoint', type=str)
    parser.add_argument('--rfr_checkpoint_save_path', type=str, default=r'C:\Users\user\Desktop\graduation\datasets\FlyingChairs\FlyingChairs_release')
    parser.add_argument('--requires_sq_flow', action='store_true')
    parser.add_argument('--model', help = 'rfr_new_gma, rfr_new_nc, rfr_new', default = 'rfr_new_nc')
    
    #settings for Normal Convolution 
    parser.add_argument('--scheduler_step', type=int, default=20000)
    parser.add_argument('--upsampler_bi', action='store_true', help='use bilinear upsampling')
    parser.add_argument('--align_corners', action='store_true', help='align_corners for bilinear upsampling')
    parser.add_argument('--freeze_raft', action='store_true', help='freeze the optical flow network and train only nc')
    parser.add_argument('--load_pretrained', default=None, help='freeze the optical flow network and train only nc')
    parser.add_argument('--compressed_ft', action='store_true', help='load the compressed version of FlyingThings3D')

    from utils.args import _add_arguments_for_module, str2bool, str2intlist
    import models.rfr_nc_model.upsampler as upsampler
    _add_arguments_for_module(
        parser,
        upsampler,
        name="final_upsampling",
        default_class='NConvUpsampler',
        exclude_classes=["_*"],
        exclude_params=["self", "args", "interpolation_net", "weights_est_net", "size"],
        param_defaults={
            "scale": 4,
            "use_data_for_guidance": True,
            "channels_to_batch": True ,
            "use_residuals": False,
            "est_on_high_res": False
        },
        forced_default_types={"scale": int,
                              "use_data_for_guidance": str2bool,
                              "channels_to_batch": str2bool,
                              "use_residuals": str2bool,
                              "est_on_high_res": str2bool},
    )

    import models.rfr_nc_model.nconv_modules as nconv_modules

    _add_arguments_for_module(
        parser,
        nconv_modules,
        name="interp_net",
        default_class='NConvUNet',
        exclude_classes=["_*"],
        exclude_params=["self", "args"],
        param_defaults={
            "channels_multiplier":2,
            "num_downsampling":1,
            "data_pooling":'conf_based',
            "encoder_fiter_sz": 5,
            "decoder_fiter_sz": 3,
            "out_filter_size": 1,
            "use_double_conv": False,
            "use_bias": False
        },
        forced_default_types={"channels_multiplier":int,
                              "num_downsampling":int,
                              "data_pooling":str,
                              "encoder_fiter_sz": int,
                               "decoder_fiter_sz": int,
                               "out_filter_size": int,
                               "use_double_conv": str2bool,
                               "use_bias": str2bool}
    )

    import models.rfr_nc_model.interp_weights_est as interp_weights_est
    _add_arguments_for_module(
        parser,
        interp_weights_est,
        name='weights_est_net',
        default_class='Simple',
        exclude_classes=["_*"],
        exclude_params=["self", "args", "out_ch", "final_act"],
        param_defaults={
            "num_ch": [64,32],
            "filter_sz": [3,3,1],
            "dilation":[1,1,1]
        },
        unknown_default_types={"num_ch": str2intlist,
                               "filter_sz": str2intlist},
        forced_default_types={"dilation": str2intlist,
                              }
    )

    #for GMA
    parser.add_argument('--upsample-learn', action='store_true', default=False,
                        help='If True, use learned upsampling, otherwise, use bilinear upsampling.')
    parser.add_argument('--position_only', default=False, action='store_true',
                        help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true',
                        help='use position and content-wise attention')
    parser.add_argument('--num_heads', default=1, type=int,
                        help='number of heads in attention and aggregation')
    

    #whole train setting
    parser.add_argument('--whole_train_epoch', type=int, default=50)
    parser.add_argument('--whole_train_checkpoint', type=str)
    parser.add_argument('--whole_train_checkpoint_epoch', type=int)
    parser.add_argument('--whole_train_model', type=str, default='AnimeInterp_no_cupy')
    parser.add_argument('--whole_train_init_learning_rate', type=float, default=1e-6)
    parser.add_argument('--whole_train_resume', action='store_true')
    parser.add_argument('--whole_train_width', type=int, default=640)
    parser.add_argument('--whole_train_height', type=int, default=360)
    parser.add_argument('--whole_train_crop', type=int, default=352)
    args = parser.parse_args()

    main(args, device)

