import os
import pickle
import numpy as np
import imageio as iio
from scipy import signal
from scipy.sparse import spdiags
import h5py

def extract_video(path: str, length: int = 900,
                  extension: str = ".png") -> np.array:
    """ Construct the video array from the png files

    Args:
        path (str): _description_
        file_str (str): _description_
        length (int, optional): _description_. Defaults to 900.
        extension (str, optional): _description_. Defaults to ".png".

    Returns:
        np.array: A 3-D array containing the video frames. [Temporal, Height, Width, Channels]
    """
    item = []
    with h5py.File(path, 'r') as f:
        video = f["images"][:]
        item.append(video)
    item = np.array(item)
    item = np.squeeze(item,axis=0)
    return item


    # video = []
    # for idx in range(length):
    #     video.append(iio.imread(os.path.join(path, f"{file_str}_{idx}{extension}")))
    # return np.array(video)

def custom_detrend(signal, Lambda):
    """custom_detrend(signal, Lambda) -> filtered_signal
    This function applies a detrending filter.
    This code is based on the following article "An advanced detrending method with application
    to HRV analysis". Tarvainen et al., IEEE Trans on Biomedical Engineering, 2002.
    *Parameters*
      ``signal`` (1d numpy array):
        The signal where you want to remove the trend.
      ``Lambda`` (int):
        The smoothing parameter.
    *Returns*
      ``filtered_signal`` (1d numpy array):
        The detrended signal.
    """
    signal_length = signal.shape[0]

    # observation matrix
    H = np.identity(signal_length)

    # second-order difference matrix

    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index, (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot((H - np.linalg.inv(H + (Lambda ** 2) * np.dot(D.T, D))), signal)
    return filtered_signal

def pulse_rate_from_power_spectral_density(pleth_sig: np.array, FS: float,
                                           LL_PR: float, UL_PR: float,
                                           BUTTER_ORDER: int = 6,
                                           DETREND: bool = False,
                                           FResBPM: float = 0.1) -> float:
    """ Function to estimate the pulse rate from the power spectral density of the plethysmography signal.

    Args:
        pleth_sig (np.array): Plethysmography signal.
        FS (float): Sampling frequency.
        LL_PR (float): Lower cutoff frequency for the butterworth filtering.
        UL_PR (float): Upper cutoff frequency for the butterworth filtering.
        BUTTER_ORDER (int, optional): Order of the butterworth filter. Give None to skip filtering. Defaults to 6.
        DETREND (bool, optional): Boolena Flag for executing cutsom_detrend. Defaults to False.
        FResBPM (float, optional): Frequency resolution. Defaults to 0.1.

    Returns:
        pulse_rate (float): _description_
    

    Daniel McDuff, Ethan Blackford, January 2019
    Copyright (c)
    Licensed under the MIT License and the RAIL AI License.
    """

    N = (60*FS)/FResBPM #Wn的范围应该要落在它之间，N计算的是频率分辨率，也就是时域的长度，这里计算出来应该是180

    # Detrending + nth order butterworth + periodogram
    if DETREND:
        pleth_sig = custom_detrend(np.cumsum(pleth_sig), 100)
    if BUTTER_ORDER:
        [b, a] = signal.butter(BUTTER_ORDER, [LL_PR/60, UL_PR/60], btype='bandpass', fs = FS)
    
    pleth_sig = signal.filtfilt(b, a, np.double(pleth_sig))
    
    # Calculate the PSD and the mask for the desired range
    F, Pxx = signal.periodogram(x=pleth_sig,  nfft=N, fs=FS);  
    FMask = (F >= (LL_PR/60)) & (F <= (UL_PR/60))
    # print(f"F{F.shape}") #(9001,) F的维度和nfft有关，所以5行信号的功率谱值共享F
    # print(f"Px{Pxx.shape}") #(5,9001)维度与输入的空间维度有关（第一维）

    # Calculate predicted pulse rate:
    FRange = F * FMask
    PRange = Pxx * FMask #心率范围
    MaxInd = np.argmax(PRange) #波峰对应的频率 返回的是最大值对应的索引
    pulse_rate_freq = FRange[MaxInd]
    pulse_rate = pulse_rate_freq*60 #频率乘60转换为心率
            
    return pulse_rate #返回的是单一的浮点数

def distribute_l_m_d(fitz_labels_path, session_names):
    # Read all the fitzpatrick labels.
    with open(fitz_labels_path, "rb") as fpf:
        out = pickle.load(fpf)
    fitz_dict = dict(out)
    l_m_d_arr = [[],[],[]]
    # Iterate over all the session names and append.
    for sess in session_names:
        pid = sess.split("_")
        sub_ix = pid[2]
        pid = pid[0] + "_" + pid[1]
        fitz_id = fitz_dict[pid]
        if(fitz_id < 3):
            l_m_d_arr[0].append(pid+'_'+sub_ix)
        elif(fitz_id < 5 and fitz_id > 2):
            l_m_d_arr[1].append(pid+'_'+sub_ix)
        else:
            l_m_d_arr[2].append(pid+'_'+sub_ix)
    return l_m_d_arr