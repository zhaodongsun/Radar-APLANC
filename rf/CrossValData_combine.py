import sys
import os
from pathlib import Path


current_file = Path(__file__).resolve()


project_root = current_file.parent.parent


sys.path.append(str(project_root))

import json
import numpy as np


def combine_and_save():

    all_est = []
    all_gt = []
    with open("nndl/rf/temp_est.json", "r") as f:
        for line in f:
            all_est.append(json.loads(line.strip()))
    with open("nndl/rf/temp_gt.json", "r") as f:
        for line in f:
            all_gt.append(json.loads(line.strip()))

    combined_list_est = []
    combined_list_gt = []
    for sublist in all_est:
        combined_list_est.extend(sublist)
    for sublist in all_gt:
        combined_list_gt.extend(sublist)

 
    final_array_est = np.array(combined_list_est)
    final_array_gt = np.array(combined_list_gt)
    final_array_est = final_array_est.flatten()
    final_array_gt = final_array_gt.flatten()

    
    np.save('nndl/rf/final_array_est.npy', final_array_est)  
    np.save('nndl/rf/final_array_gt.npy', final_array_gt)


    print(f"est shape: {final_array_est.shape}")
    print(f'gt shape: {final_array_gt.shape}')


if __name__ == "__main__":
    combine_and_save()