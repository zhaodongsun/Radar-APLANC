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
from rf.IQ_to_PhaseAngle import IQ_to_PhaseAngle

import matplotlib.pyplot as plt
import rf.organizer as org
from utils_sig import *
from rf.proc import create_fast_slow_matrix, find_range

from utils.utils import extract_video, pulse_rate_from_power_spectral_density
from scipy.signal import resample_poly
from scipy.signal import detrend






class RHB(Dataset):
    def __init__(self, datapath, datapaths, ppg_signal_length=900, frame_length_ppg=512, sampling_ratio=4, \
                 window_size=5, samples=256, samp_f=5e6, freq_slope=60.012e12,Pdatapath='APLANC/dataset/Pseudo_Labels_RHB') -> None:

   
        self.ppg_offset = 0
        # Data structure for videos.
        self.datapath = datapath
        # Load videos and signals.
        self.rf_file_list = datapaths  # [1_1,2_5,........]
        self.Pseudo_list = []
        # Load signals.
        # remove_list_folder = []
        for folder in self.rf_file_list:
            file_path = os.path.join(Pdatapath, folder)
            if (os.path.exists(os.path.join(file_path, "pseudo.npy"))):
                Pseudo = np.load(f"{file_path}/pseudo.npy",allow_pickle=True)
                self.Pseudo_list.append(Pseudo[0:900])
            else:
                print(f"{folder} not found.")
            # else:
            #     remove_list_folder.append(folder)

       

        # The ratio of the sampling frequency of the RF signal and the PPG signal.
        self.sampling_ratio = sampling_ratio  # 4

        # Save the RF config parameters.
        self.window_size = window_size  # 5
        self.samples = samples  # 256
        self.samp_f = samp_f  # 5e6
        self.freq_slope = freq_slope  # 60.012e12 

        # Window the PPG and the RF samples.
        self.ppg_signal_length = ppg_signal_length  # 900
        self.frame_length_ppg = frame_length_ppg  # 512
        

    
        # High-ram, compute FFTs before starting training.
        self.rf_data_p_list = []
        self.rf_data_n_list = []
        self.traditional_list = []
        # self.pseudo_filter = PositiveFiliter(2,120,45,150,0.1,None)
        for rf_file in self.rf_file_list:
            # Read the raw RF data
            rf_fptr = open(os.path.join(self.datapath, rf_file, "rf.pkl"), 'rb')
            s = pickle.load(rf_fptr)
           
            rf_organizer = org.Organizer(s, 1, 1, 1, 2 * self.samples) 
          
            frames = rf_organizer.organize()
            # The RF read adds zero alternatively to the samples. Remove these zeros.
            frames = frames[:, :, :, 0::2]  # (num_frames,num_chirps,num_rx,num_samples) （3599，1，1，256）

            # Process the organized RF data
            data_f = create_fast_slow_matrix(frames)  # range matrix(fre domain) (num_frames,num_samples) （3599,256）
            range_index = find_range(data_f, self.samp_f, self.freq_slope, self.samples) 
            # Get the windowed raw data for the network
            raw_data_p = data_f[:,
                         range_index - self.window_size // 2:range_index + self.window_size // 2 + 1] 
            raw_data_left = data_f[:, 0:range_index - self.window_size // 2 - 1]  
            # raw_data_right = data_f[:,range_index+self.window_size//2+1:] 
            raw_data_right = data_f[:, range_index + self.window_size // 2 + 30:]  
            Phase_data_p = IQ_to_PhaseAngle(raw_data_p,0.8,2.5,120)
            Phase_data_p = sig.decimate(Phase_data_p,4)
            if raw_data_left.shape[1] > raw_data_right.shape[1]:
                n_samples_start = np.random.choice(raw_data_left.shape[1] - self.window_size)
                raw_data_n = raw_data_left[:, n_samples_start:n_samples_start + self.window_size]
                
            else:
                n_samples_start = np.random.choice(raw_data_right.shape[1] - self.window_size)
                raw_data_n = raw_data_right[:, n_samples_start:n_samples_start + self.window_size]
              

            raw_data_p = np.array(
                [np.real(raw_data_p), np.imag(raw_data_p)])  
            raw_data_n = np.array([np.real(raw_data_n), np.imag(raw_data_n)])
      

            raw_data_p = np.transpose(raw_data_p, axes=(0, 2, 1))  
            raw_data_n = np.transpose(raw_data_n, axes=(0, 2, 1)) 
            self.rf_data_n_list.append(raw_data_n) 
            self.rf_data_p_list.append(raw_data_p) 
            self.traditional_list.append(Phase_data_p)

    def __len__(self):
        return len(self.rf_file_list)

    def __getitem__(self, idx):
    

        # Get the RF data.
    
        p_data_f = self.rf_data_p_list[idx] 
        n_data_f = self.rf_data_n_list[idx]
        traditional_data_f = self.traditional_list[idx]
        some_data_p = p_data_f[:,:,0:100]
        some_data_n = n_data_f[:,:,0:100]
        some_data_t = traditional_data_f[:,0:100]
        p_data_f = np.concatenate((p_data_f, some_data_p), axis=2)
        n_data_f = np.concatenate((n_data_f, some_data_n), axis=2)
        traditional_data_f = np.concatenate((traditional_data_f, some_data_t), axis=1)
      
   
        frame_start = 0
     

        p_data_f = p_data_f[:, :,
                   frame_start * self.sampling_ratio:frame_start * self.sampling_ratio + self.sampling_ratio * self.frame_length_ppg]
        n_data_f = n_data_f[:, :,
                   frame_start * self.sampling_ratio:frame_start * self.sampling_ratio + self.sampling_ratio * self.frame_length_ppg]
        traditional_data_f = traditional_data_f[2,frame_start:frame_start + self.frame_length_ppg]


  
        Pseudo = self.Pseudo_list[idx][frame_start:frame_start + self.frame_length_ppg]
        assert len(
            Pseudo) == self.frame_length_ppg, f"Expected signal of length {self.frame_length_ppg}, but got signal of length {len(Pseudo)}"

        return  Pseudo,p_data_f, n_data_f,traditional_data_f



class Equi(Dataset):
    def __init__(self, datapath, datapaths, ppg_signal_length=900, frame_length_ppg=512, sampling_ratio=4, \
                 window_size=5, samples=256, samp_f=5e6, freq_slope=60.012e12,Pdatapath='APLANC/dataset/Pseudo_Labels_Equi') -> None:

        # There is an offset in capturing the signals w.r.t the ground truth.
        self.ppg_offset = 0
        # Number of samples to be created by oversampling one trial.
        # self.num_samps = static_dataset_samples

        # Data structure for videos.
        self.datapath = datapath
        # Load videos and signals.
        self.rf_file_list = datapaths  # [1_1,2_5,........]
        self.Pseudo_list = []
        # Load signals.
        # remove_list_folder = []
        for folder in self.rf_file_list:
            file_path = os.path.join(Pdatapath, folder)
            if (os.path.exists(os.path.join(file_path, "pseudo.npy"))):
            
                Pseudo = np.load(f"{file_path}/pseudo.npy",allow_pickle=True)
                self.Pseudo_list.append(Pseudo[0:900])
            else:
                print(f"{folder} not found.")
            # else:
            #     remove_list_folder.append(folder)


        # The ratio of the sampling frequency of the RF signal and the PPG signal.
        self.sampling_ratio = sampling_ratio  # 4

        # Save the RF config parameters.
        self.window_size = window_size  # 5
        self.samples = samples  # 256
        self.samp_f = samp_f  # 5e6
        self.freq_slope = freq_slope  # 60.012e12 #频率的斜率

        # Window the PPG and the RF samples.
        self.ppg_signal_length = ppg_signal_length  # 900
        self.frame_length_ppg = frame_length_ppg  # 512
     

        # 获取windowed rf signal
        # High-ram, compute FFTs before starting training.
        self.rf_data_p_list = []
        self.rf_data_n_list = []
        self.traditional_list = []
        # self.pseudo_filter = PositiveFiliter(2,120,45,150,0.1,None)
        for rf_file in self.rf_file_list:
            # Read the raw RF data
            rf_fptr = open(os.path.join(self.datapath, rf_file, "rf.pkl"), 'rb')
            s = pickle.load(rf_fptr)
        
            rf_organizer = org.Organizer(s, 1, 1, 1, 2 * self.samples)  
       
            frames = rf_organizer.organize()
            # The RF read adds zero alternatively to the samples. Remove these zeros.
            frames = frames[:, :, :, 0::2] 

            # Process the organized RF data
            data_f = create_fast_slow_matrix(frames) 
            range_index = find_range(data_f, self.samp_f, self.freq_slope, self.samples) 
            # Get the windowed raw data for the network
            raw_data_p = data_f[:,
                         range_index - self.window_size // 2:range_index + self.window_size // 2 + 1]  
            raw_data_left = data_f[:, 0:range_index - self.window_size // 2 - 1]  
            # raw_data_right = data_f[:,range_index+self.window_size//2+1:] 
            raw_data_right = data_f[:, range_index + self.window_size // 2 + 30:]  
            Phase_data_p = IQ_to_PhaseAngle(raw_data_p,0.8,2.5,120) 
            Phase_data_p = sig.decimate(Phase_data_p,4)
            if raw_data_left.shape[1] > raw_data_right.shape[1]:
                n_samples_start = np.random.choice(raw_data_left.shape[1] - self.window_size)
                raw_data_n = raw_data_left[:, n_samples_start:n_samples_start + self.window_size]
                # Phase_data_n = IQ_to_PhaseAngle(raw_data_n)
            else:
                n_samples_start = np.random.choice(raw_data_right.shape[1] - self.window_size)
                raw_data_n = raw_data_right[:, n_samples_start:n_samples_start + self.window_size]
                # Phase_data_n = IQ_to_PhaseAngle(raw_data_n)

            raw_data_p = np.array(
                [np.real(raw_data_p), np.imag(raw_data_p)])  
            raw_data_n = np.array([np.real(raw_data_n), np.imag(raw_data_n)])
           
            raw_data_p = np.transpose(raw_data_p, axes=(0, 2, 1))  
            raw_data_n = np.transpose(raw_data_n, axes=(0, 2, 1))  
            self.rf_data_n_list.append(raw_data_n) 
            self.rf_data_p_list.append(raw_data_p) 
            self.traditional_list.append(Phase_data_p)

    def __len__(self):
        return len(self.rf_file_list)

    def __getitem__(self, idx):
     

       
        p_data_f = self.rf_data_p_list[idx]  # (num_samples,num_frames)(D,T)
        n_data_f = self.rf_data_n_list[idx]
        traditional_data_f = self.traditional_list[idx]
        some_data_p = p_data_f[:,:,0:100]
        some_data_n = n_data_f[:,:,0:100]
        some_data_t = traditional_data_f[:,0:100]
        p_data_f = np.concatenate((p_data_f, some_data_p), axis=2)
        n_data_f = np.concatenate((n_data_f, some_data_n), axis=2)
        traditional_data_f = np.concatenate((traditional_data_f, some_data_t), axis=1)
       
        frame_start = 0
   
        # data_f = data_f[:,:,rf_start : rf_start + (self.sampling_ratio * self.frame_length_ppg)] #(2, num_frames, num_samples)
        p_data_f = p_data_f[:, :,
                   frame_start * self.sampling_ratio:frame_start * self.sampling_ratio + self.sampling_ratio * self.frame_length_ppg]
        n_data_f = n_data_f[:, :,
                   frame_start * self.sampling_ratio:frame_start * self.sampling_ratio + self.sampling_ratio * self.frame_length_ppg]
      
        traditional_data_f = traditional_data_f[1,frame_start:frame_start + self.frame_length_ppg]

        # print(p_data_f.shape)

        # item = data_f

        # Get the PPG signal.
        Pseudo = self.Pseudo_list[idx][frame_start:frame_start + self.frame_length_ppg]
        assert len(
            Pseudo) == self.frame_length_ppg, f"Expected signal of length {self.frame_length_ppg}, but got signal of length {len(Pseudo)}"

        # return item, np.array(item_sig)
        return  Pseudo,p_data_f, n_data_f,traditional_data_f
