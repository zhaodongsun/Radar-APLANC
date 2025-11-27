import sys
import os
from pathlib import Path


current_file = Path(__file__).resolve()


project_root = current_file.parent.parent


sys.path.append(str(project_root))
import os
import pickle
import numpy as np
import imageio
import scipy.signal as sig
from fontTools.t1Lib import decryptType1
from numpy.array_api import uint16
from numpy.ma.core import array
from torch.utils.data import Dataset
import sys
import h5py
import torch
from rf.IQ_to_PhaseAngle import IQ_to_PhaseAngle

import matplotlib.pyplot as plt
import rf.organizer as org
from utils_sig import *
from utils.errors import getErrors
from rf.proc import create_fast_slow_matrix, find_range
from utils.utils import pulse_rate_from_power_spectral_density
from rf.model import RF_conv_decoder

def min(model,data_for_Pred,Pseudo_Labels,frame_length):
    cur_est_ppgs = None
    est_p = None
    for cur_frame_num in range(data_for_Pred.shape[0]):

        cur_frame = data_for_Pred[cur_frame_num, :, :]  
        cur_frame = torch.tensor(cur_frame).type(torch.float32)
        cur_frame = normalize(cur_frame)
 
        cur_frame = cur_frame.unsqueeze(0).to('cuda') 
   
        if cur_frame_num % (frame_length) == 0:
            cur_cat_frames = cur_frame
        else:
            cur_cat_frames = torch.cat((cur_cat_frames, cur_frame), 0)

       
        if cur_cat_frames.shape[0] == frame_length: 

       
            with torch.no_grad():
            
                cur_cat_frames = cur_cat_frames.unsqueeze(0) 
                cur_cat_frames = torch.transpose(cur_cat_frames, 1, 2)  
                cur_cat_frames = torch.transpose(cur_cat_frames, 2, 3)  
             
                IQ_frames = torch.reshape(cur_cat_frames,(cur_cat_frames.shape[0], -1, cur_cat_frames.shape[3]))  # (1,10,512)
                cur_est_ppg, _ = model(IQ_frames) 
                cur_est_ppg = cur_est_ppg.squeeze().cpu().numpy()  

          
            if cur_est_ppgs is None:
                cur_est_ppgs = cur_est_ppg
            else:
                cur_est_ppgs = np.concatenate((cur_est_ppgs, cur_est_ppg), -1)
    est_ppgs = cur_est_ppgs[0:900]
    pseudo_labels = Pseudo_Labels[:,0:900]
    est_p = est_ppgs[0:900]

    hr_window_size = 300
    stride = 300

    mae_list = [] 
    hr_est_temp = []

     
    est_ppgs = (est_ppgs-np.mean(est_ppgs))/np.std(est_ppgs)
    for start in range(0, len(est_ppgs) - hr_window_size, stride):
        ppg_est_window = est_ppgs[start:start + hr_window_size]
  
        ppg_est_window = (ppg_est_window - np.mean(ppg_est_window)) / np.std(ppg_est_window)
        hr_est_temp.append(pulse_rate_from_power_spectral_density(
                ppg_est_window, 30, 45, 150, BUTTER_ORDER=6, DETREND=False))

    hr_est_windowed = np.array([hr_est_temp])
    
    for pseudo_label in range(pseudo_labels.shape[0]):
        cur_pseudo_label = pseudo_labels[pseudo_label,:]
        cur_pseudo_label = (cur_pseudo_label - np.mean(cur_pseudo_label)) / np.std(cur_pseudo_label)
        hr_pseudo_temp = []
        for start in range(0, len(cur_pseudo_label) - hr_window_size, stride):
            ppg_pseudo_window = cur_pseudo_label[start:start + hr_window_size]
            ppg_pseudo_window = (ppg_pseudo_window-np.mean(ppg_pseudo_window)) / np.std(ppg_pseudo_window)
            hr_pseudo_temp.append(pulse_rate_from_power_spectral_density(
                    ppg_pseudo_window, 30, 45, 150, BUTTER_ORDER=6, DETREND=False))
        hr_pseudo_windowed = np.array(hr_pseudo_temp)
        _, MAE, _, _ = getErrors(hr_est_windowed, hr_pseudo_windowed)
        mae_list.append(MAE)
    mae_list = torch.tensor(mae_list)
    index = torch.argmin(mae_list)
    print(f'P_MAE_list: {mae_list},Choose_index: {index}')

    return index,est_p

def max(model,data_for_Pred,Pseudo_Labels,frame_length):
    cur_est_ppgs = None
    est_n = None
    for cur_frame_num in range(data_for_Pred.shape[0]):

        cur_frame = data_for_Pred[cur_frame_num, :, :]  
        cur_frame = torch.tensor(cur_frame).type(torch.float32)
        cur_frame = normalize(cur_frame)
   
        cur_frame = cur_frame.unsqueeze(0).to('cuda')  

        if cur_frame_num % (frame_length) == 0:
            cur_cat_frames = cur_frame
        else:
            cur_cat_frames = torch.cat((cur_cat_frames, cur_frame), 0)

        if cur_cat_frames.shape[0] == frame_length:  

        
            with torch.no_grad():
             
                cur_cat_frames = cur_cat_frames.unsqueeze(0)  
                cur_cat_frames = torch.transpose(cur_cat_frames, 1, 2) 
                cur_cat_frames = torch.transpose(cur_cat_frames, 2, 3) 
        
                IQ_frames = torch.reshape(cur_cat_frames,(cur_cat_frames.shape[0], -1, cur_cat_frames.shape[3]))  
                cur_est_ppg, _ = model(IQ_frames) 
                cur_est_ppg = cur_est_ppg.squeeze().cpu().numpy()  


            if cur_est_ppgs is None:
                cur_est_ppgs = cur_est_ppg
            else:
                cur_est_ppgs = np.concatenate((cur_est_ppgs, cur_est_ppg), -1)
    est_ppgs = cur_est_ppgs[0:900]
    pseudo_labels = Pseudo_Labels[:,0:900]
    est_n = est_ppgs[0:900]

    hr_window_size = 300
    stride = 300
    mae_list = [] 
    hr_est_temp = []


    est_ppgs = (est_ppgs-np.mean(est_ppgs))/np.std(est_ppgs)
    for start in range(0, len(est_ppgs) - hr_window_size, stride):
        ppg_est_window = est_ppgs[start:start + hr_window_size]
   
        ppg_est_window = (ppg_est_window - np.mean(ppg_est_window)) / np.std(ppg_est_window)
        hr_est_temp.append(pulse_rate_from_power_spectral_density(
                ppg_est_window, 30, 45, 150, BUTTER_ORDER=6, DETREND=False))

    hr_est_windowed = np.array([hr_est_temp])

    for pseudo_label in range(pseudo_labels.shape[0]):
        cur_pseudo_label = pseudo_labels[pseudo_label,:]
        cur_pseudo_label = (cur_pseudo_label - np.mean(cur_pseudo_label)) / np.std(cur_pseudo_label)
        hr_pseudo_temp = []
        for start in range(0, len(cur_pseudo_label) - hr_window_size, stride):
            ppg_pseudo_window = cur_pseudo_label[start:start + hr_window_size]
            ppg_pseudo_window = (ppg_pseudo_window-np.mean(ppg_pseudo_window)) / np.std(ppg_pseudo_window)
            hr_pseudo_temp.append(pulse_rate_from_power_spectral_density(
                    ppg_pseudo_window, 30, 45, 150, BUTTER_ORDER=6, DETREND=False))
        hr_pseudo_windowed = np.array(hr_pseudo_temp)
        _, MAE, _, _ = getErrors(hr_est_windowed, hr_pseudo_windowed)
        mae_list.append(MAE)
    mae_list = torch.tensor(mae_list)
    index = torch.argmax(mae_list)
    print(f'N_MAE_list: {mae_list},Choose_index: {index}')

    return index,est_n,mae_list

def NMinusP(est_n,est_p):
    hr_window_size = 300
    stride = 300
    mae_list = [] 
    hr_est_temp_p = []
    hr_est_temp_n = []

    est_n = (est_n - np.mean(est_n)) / np.std(est_n)
    est_p = (est_p - np.mean(est_p)) / np.std(est_p)

    assert len(est_n) == len(est_p), f"length of est_n{len(est_n)} != length of est_p{len(est_p)}"

    for start in range(0, len(est_n) - hr_window_size, stride):
        ppg_est_window_n = est_n[start:start + hr_window_size]
        ppg_est_window_p = est_p[start:start + hr_window_size]

        ppg_est_window_n = (ppg_est_window_n - np.mean(ppg_est_window_n)) / np.std(ppg_est_window_n)
        ppg_est_window_p = (ppg_est_window_p - np.mean(ppg_est_window_p)) / np.std(ppg_est_window_p)
        hr_est_temp_n.append(pulse_rate_from_power_spectral_density(
                ppg_est_window_n, 30, 45, 150, BUTTER_ORDER=6, DETREND=False))
        hr_est_temp_p.append(pulse_rate_from_power_spectral_density(
            ppg_est_window_p, 30, 45, 150, BUTTER_ORDER=6, DETREND=False))
    hr_est_windowed_n = np.array([hr_est_temp_n])
    hr_est_windowed_p = np.array(hr_est_temp_p)
    _,MAE,_,_ = getErrors(hr_est_windowed_n,hr_est_windowed_p)
    print(f'NMP_MAE:{MAE}')
    return MAE



def Pseudo_Label_Generator(cpkt_path,datapath, datapaths,save_root, window_size=5, samples=256, samp_f=5e6, freq_slope=60.012e12):
 
        model_P = RF_conv_decoder()
        model_N = RF_conv_decoder()
        model_P.load_state_dict(torch.load(cpkt_path)['model_P_state_dict'])
        model_N.load_state_dict(torch.load(cpkt_path)['model_N_state_dict'])
        model_P.eval()
        model_N.eval()
        model_P.to('cuda')
        model_N.to('cuda')

        rf_file_list = datapaths


        traditional_list = []
        for rf_file in rf_file_list:
   
            rf_fptr = open(os.path.join(datapath, rf_file, "rf.pkl"), 'rb')
            s = pickle.load(rf_fptr)
            rf_organizer = org.Organizer(s, 1, 1, 1, 2 * samples)  
  
            frames = rf_organizer.organize()
    
            frames = frames[:, :, :, 0::2]  

            data_f = create_fast_slow_matrix(frames)  
            range_index = find_range(data_f, samp_f, freq_slope, samples) 

            raw_data_p = data_f[:,range_index - window_size // 2:range_index + window_size // 2 + 1] 
            data_p = np.array([np.real(raw_data_p),np.imag(raw_data_p)])
            data_p = np.transpose(data_p, axes=(0, 2, 1))
            data_p = np.transpose(data_p, axes=(2, 0, 1)) 
            Phase_data_p = IQ_to_PhaseAngle(raw_data_p,0.8,2.5,120)
            Phase_data_p = sig.decimate(Phase_data_p,4)

            raw_data_left = data_f[:, 0:range_index - 5 // 2 - 1]  
            raw_data_right = data_f[:, range_index + 5 // 2 + 30:] 
            if raw_data_left.shape[1] > raw_data_right.shape[1]:
                n_samples_start = np.random.choice(raw_data_left.shape[1] - 5)
                raw_data_n = raw_data_left[:, n_samples_start:n_samples_start + 5]
            else:
                n_samples_start = np.random.choice(raw_data_right.shape[1] - 5)
                raw_data_n = raw_data_right[:, n_samples_start:n_samples_start + 5]
            raw_data_n = np.array([np.real(raw_data_n), np.imag(raw_data_n)])
            data_n = np.transpose(raw_data_n, axes=(0, 2, 1))  
            data_n = np.transpose(data_n, axes=(2, 0, 1))  

            circ = data_p[0:100,:,:]
            Pcirc = Phase_data_p[:,0:100]
            circ_n = data_n[0:100,:,:]
            data_n = np.concatenate((circ_n,data_n),axis=0)
            data_p = np.concatenate((circ,data_p),axis=0)
            Phase_data_p = np.concatenate((Pcirc,Phase_data_p),axis=1) 


            index_p,est_p = min(model_P,data_p,Phase_data_p,frame_length=3600)
            index_n,est_n,mae_list_N = max(model_N,data_n,Phase_data_p,frame_length=3600)
            Vf = NMinusP(est_n,est_p)
            pseudo_noise_mae = mae_list_N[index_p]

            if index_p == index_n:
                os.makedirs(os.path.join(save_root, rf_file), exist_ok=True)
                np.save(os.path.join(save_root, rf_file, "pseudo.npy"), Phase_data_p[index_p])
                print(f'{index_p}=={index_n,}Finally Choose index {index_p}')
            elif pseudo_noise_mae.item() < Vf.item():
                est_p = butter_bandpass(est_p,0.8,2.5,30)
                os.makedirs(os.path.join(save_root, rf_file), exist_ok=True)
                np.save(os.path.join(save_root, rf_file, "pseudo.npy"), est_p)
                print(f'pseudos are bad,Pseudo to Noise:{pseudo_noise_mae.item()}<Est to noise:{Vf.item()},choose filtered est feature!')
            else:
                os.makedirs(os.path.join(save_root, rf_file), exist_ok=True)
                np.save(os.path.join(save_root, rf_file, "pseudo.npy"), Phase_data_p[index_p])
                print(f'best index:{index_p},Still Choose index {index_p}')

        print("Finish")


def main():
    datapath = 'APLANC/dataset/rf_files'
    folds_path = 'APLANC/dataset/Equi_demo_fold.pkl'
    save_root = 'APLANC/dataset/Pseudo_Labels_Equi'
    with open(folds_path, "rb") as fp:
        files_in_fold = pickle.load(fp)
    train_files = files_in_fold[0]["train"]
    train_files = [i[2:] for i in train_files]


    cpkt_path = 'APLANC/rf/ckpt/Equi/best.pth'

    Pseudo_Label_Generator(cpkt_path,datapath=datapath,datapaths=train_files,save_root=save_root)

if __name__ == '__main__':
    main()