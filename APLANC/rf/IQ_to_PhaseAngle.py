import numpy as np
from utils_sig import *

def IQ_to_PhaseAngle(IQ_data,lowcut,highcut,fs):
    unwrap_angle = []
    IQ_data = np.angle(IQ_data)
    for i in range(IQ_data.shape[1]): 
        cur_phase = np.unwrap(IQ_data[:,i])#(3599,) 
        cur_phase = butter_bandpass(cur_phase,lowcut=lowcut,highcut=highcut,fs=fs)
        unwrap_angle.append(cur_phase)

    return np.array(unwrap_angle)