import sys
import os
from pathlib import Path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root))
import numpy as np
import pickle
import os
import argparse
import matplotlib.pyplot as plt

import torch

from rf.model import RF_conv_decoder
from utils.eval import demo_eval,eval_trad,eval_our_rf, eval_performance_bias, eval_clinical_performance

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# Argparser.
def parseArgs():
    parser = argparse.ArgumentParser(description='Configs for thr RF test script')

    parser.add_argument('-dir', '--data-dir', default="APLANC/dataset/RHB", type=str,
                        help="Parent directory containing the folders with the PNG images and the PPG npy.")

    parser.add_argument('--fold', type=int, default=0,
                        help='Fold Number')

    parser.add_argument('--device', type=str, default=None,
                        help="Device on which the model needs to run (input to torch.device). \
                              Don't specify for automatic selection. Will be modified inplace.")

    parser.add_argument('-ckpt', '--checkpoint-path', type=str,
                        default="APLANC/rf/ckpt/OurRF/best.pth",
                        help='Checkpoint Folder.')

    parser.add_argument('--verbose', action='store_true', help="Verbosity.")

    parser.add_argument('--viz', action='store_true', help="Visualize.")

    return parser.parse_args()


def main(args):
    # Import essential info, i.e. destination folder and fitzpatrick label path
    destination_folder = args.data_dir
    ckpt_path = args.checkpoint_path


    
    test_files = ['1_1','2_1','3_1']

    # Select the device
    if args.device is None:
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device(args.device)
    if args.verbose:
        print('Running on device: {}'.format(args.device))

    model = RF_conv_decoder().to(args.device)

    model.load_state_dict(torch.load(ckpt_path, map_location='cuda')['model_P_state_dict'])

    maes_test, hr_test, video_samples = demo_eval(destination_folder, test_files, model, device=args.device,flag=True)


if __name__ == '__main__':
    args = parseArgs()
    main(args)