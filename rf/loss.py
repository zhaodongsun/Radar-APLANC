import torch
import torch.nn as nn


tr = torch
import torch.nn.functional as F
import numpy as np
import torch.fft

class  ContrastLoss(nn.Module):
    def __init__(self,delta_t, K,Fs, high_pass, low_pass):
        super(ContrastLoss, self).__init__()
        self.RF_sampling = RF_sampling(delta_t,K,Fs, high_pass, low_pass) # spatiotemporal sampler
        self.Pseudo_sampling = RF_sampling(delta_t*4,K,Fs*4, high_pass, low_pass)
        self.distance_func = nn.MSELoss(reduction = 'mean') # mean squared error for comparing two PSDs
        # self.distance_func = nn.L1Loss(reduction = 'mean')

    def compare_samples(self, list_a, list_b, exclude_same=False):
        if exclude_same:
            total_distance = 0.
            M = 0
            for i in range(len(list_a)):
                for j in range(len(list_b)):
                    if i != j:
                        total_distance += self.distance_func(list_a[i], list_b[j])
                        M += 1
        else:
            total_distance = 0.
            M = 0
            for i in range(len(list_a)):
                for j in range(len(list_b)):
                    total_distance += self.distance_func(list_a[i], list_b[j])
                    M += 1

        return total_distance / M

    def forward(self, model_output_p,model_output_n,gt_signal):
        samples_p = self.RF_sampling(model_output_p) #dim = [B,T]
        samples_n = self.RF_sampling(model_output_n)
        gt_signal = self.RF_sampling(gt_signal)
   


        extra_pos_loss = (self.compare_samples(samples_p[0],samples_p[0],exclude_same=True) + self.compare_samples(samples_p[1],samples_p[1],exclude_same=True))/2
        extra_neg_loss = -self.compare_samples(samples_p[0],samples_p[1])

        extra_loss = extra_pos_loss
   

        # positive loss 
        pos_loss = (self.compare_samples(samples_p[0], gt_signal[0]) + self.compare_samples(samples_p[1], gt_signal[1])
                   ) / 2
        # negative loss
        neg_loss = -(self.compare_samples(samples_p[0], samples_n[0]) + self.compare_samples(samples_p[1],samples_n[1])
                      ) / 2

      
        loss = pos_loss + neg_loss

     

       
        return loss, pos_loss, neg_loss,extra_loss


class RF_sampling(nn.Module):
    
    def __init__(self,delta_t, K,Fs, high_pass, low_pass):
        super().__init__()
        self.delta_t = delta_t
        self.K = K
        self.norm_psd = CalculateNormPSD(Fs, high_pass, low_pass)


    def forward(self, input): # input: (B, chirps)
        samples = []
        for b in range(input.shape[0]): # loop over videos (totally 2 videos)
            samples_per_rf = []
            for i in range(self.K):
                offset = torch.randint(0, input.shape[-1] - self.delta_t + 1, (1,), device=input.device)
                # print(input[b,offset:offset+self.delta_t])
                x = self.norm_psd(input[b,offset:offset+self.delta_t])
                samples_per_rf.append(x)
            samples.append(samples_per_rf)
        return samples


class CalculateNormPSD(nn.Module):
    # we reuse the code in Gideon2021 to get the normalized power spectral density
    # Gideon, John, and Simon Stent. "The way to my heart is through contrastive learning: Remote photoplethysmography from unlabelled video." Proceedings of the IEEE/CVF international conference on computer vision. 2021.
    
    def __init__(self, Fs, high_pass, low_pass):
        super().__init__()
        self.Fs = Fs
        self.high_pass = high_pass
        self.low_pass = low_pass

    def forward(self, x, zero_pad=0):
        x = x - torch.mean(x, dim=-1, keepdim=True)
        if zero_pad > 0:
            L = x.shape[-1]
            x = F.pad(x, (int(zero_pad/2*L), int(zero_pad/2*L)), 'constant', 0)

        # Get PSD
        x = torch.view_as_real(torch.fft.rfft(x, dim=-1, norm='forward'))
        x = tr.add(x[:, 0] ** 2, x[:, 1] ** 2)

        # Filter PSD for relevant parts
        Fn = self.Fs / 2
        freqs = torch.linspace(0, Fn, x.shape[0])
        use_freqs = torch.logical_and(freqs >= self.high_pass / 60, freqs <= self.low_pass / 60) #这里对psd做了滤波
        x = x[use_freqs]

        # Normalize PSD
        x = x / torch.sum(x, dim=-1, keepdim=True)
        return x