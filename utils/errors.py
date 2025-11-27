import numpy as np
from numpy.ma.core import absolute
from scipy import stats



def getErrors(bpmES, bpmGT, timesES=None, timesGT=None, PCC=True):
    RMSE = RMSEerror(bpmES, bpmGT, timesES, timesGT)
    MAE = MAEerror(bpmES, bpmGT, timesES, timesGT)
    MAX = MAXError(bpmES, bpmGT, timesES, timesGT)
    if(PCC == True):
        PCC = PearsonCorr(bpmES, bpmGT, timesES, timesGT)
    else:
        PCC = None
    return RMSE, MAE, MAX, PCC

def RMSEerror(bpmES, bpmGT, timesES=None, timesGT=None):
    """ RMSE: """

    diff = bpm_diff(bpmES, bpmGT, timesES, timesGT)
    n,m = diff.shape  # n = num channels, m = bpm length
    df = np.zeros(n)
    for j in range(m):
        for c in range(n):
            df[c] += np.power(diff[c,j],2)

    # -- final RMSE
    RMSE = np.sqrt(df/m)
    return RMSE

def MAEerror(bpmES, bpmGT, timesES=None, timesGT=None):
    """ MAE: """

    diff = bpm_diff(bpmES, bpmGT, timesES, timesGT)
    n,m = diff.shape  # n = num channels, m = bpm length(1,5)
    df = np.sum(np.abs(diff),axis=1)

    # -- final MAE
    MAE = df/m
    return MAE

def MAXError(bpmES, bpmGT, timesES=None, timesGT=None):
    """ MAE: """

    diff = bpm_diff(bpmES, bpmGT, timesES, timesGT)
    n,m = diff.shape  # n = num channels, m = bpm length
    # print(diff.shape)
    df = np.max(np.abs(diff),axis=1)

    # -- final MAE
    MAX = df
    return MAX

def PearsonCorr(bpmES, bpmGT, timesES=None, timesGT=None):

    diff = bpm_diff(bpmES, bpmGT, timesES, timesGT)
    n,m = diff.shape  # n = num channels, m = bpm length
    CC = np.zeros(n)
    for c in range(n):
        # -- corr
        r,p = stats.pearsonr(diff[c,:]+bpmES[c,:],bpmES[c,:])
        CC[c] = r
    return CC

def bpm_diff(bpmES, bpmGT, timesES=None, timesGT=None):
    n,m = bpmES.shape  # n = num channels, m = bpm length
    # print(f"bpsES的shape:{bpmES.shape},bpmGT的shape:{bpmGT.shape}")
    if (timesES is None) or (timesGT is None):
        timesES = np.arange(m)
        timesGT = timesES
            
    diff = np.zeros((n,m)) #(1,5)
    for j in range(m):
        t = timesES[j]
        i = np.argmin(np.abs(t-timesGT))
        for c in range(n):
            diff[c,j] = bpmGT[i]-bpmES[c,j]
    # print(f'同一样本不同窗口的MAE差值:{diff}')
    return diff #这以上代码其实就只是在让同索引的心率相减，也就是同一窗口的心率 返回的差值为m个，n维度一样为1 i.e.[[1,2,3,4,5]] array

def printErrors(RMSE, MAE, MAX, PCC):
    print("\n    * Errors: RMSE = %.2f, MAE = %.2f, MAX = %.2f, PCC = %.2f" %(RMSE,MAE,MAX,PCC))