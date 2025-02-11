import numpy as np
import os.path as osp
import pickle
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq

class task_2_2:
    def __init__(self, data_root="./data/") -> None:
        """
        Initializes the task_2_2 class, loading various signal data from pickle files.

        Attributes:
            data_root (str): The root directory where data files are stored.
            spt_fn (str): Filename for the sum of pure tone signals (task_2_2_1).
            chirp_fn (str): Filename for the chirp signal (task_2_2_2).
            ecg_fn (str): Filename for the ECG signal (task_2_2_3).
            spt_data (dict): Loaded data for the sum of pure tone signals.
            chirp_data (dict): Loaded data for the chirp signal.
            ecg_data (dict): Loaded data for the ECG signal.
        """
        self.data_root = data_root
        self.spt_fn = "task_2_2_1.pickle"
        self.chirp_fn = "task_2_2_2.pickle"
        self.ecg_fn = "task_2_2_3.pickle"
        
        with open(osp.join(self.data_root, self.spt_fn), "rb") as f:
            self.spt_data = pickle.load(f)
        with open(osp.join(self.data_root, self.chirp_fn), "rb") as f:
            self.chirp_data = pickle.load(f)
        with open(osp.join(self.data_root, self.ecg_fn), "rb") as f:
            self.ecg_data = pickle.load(f)
        
    def get_freq_spt(self):
        """
        Analyze the sum of pure tone signals to determine the primary frequency components.

        Returns:
            freq (np.float64): An array of the three primary frequency components in descending order.
        
        >>> test = task_2_2()
        >>> f = test.get_freq_spt()
        >>> len(f) == 3
        True
        """
        s_t = self.spt_data["values"] # signal values
        fs = self.spt_data["fs"] # sampling frequency
        
        freq = np.zeros(3, dtype=np.float64) # (3,)
        
        # >>>>>>>>>>>>>>> YOUR CODE HERE <<<<<<<<<<<<<<<
        # TODO:
        # >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<
        s_f = fft(s_t)
        s_f = np.abs(s_f)
        s_f_freq = fftfreq(len(s_t), 1/fs)
        n = len(s_f)
        s_f = s_f[:n//2]
        s_f_freq = s_f_freq[:n//2]
        # plt.plot(s_f_freq, s_f)
        # plt.show()
        freq, _ = find_peaks(s_f)
        freq = s_f[freq]
        return freq
    
    def get_bw_chirp(self):
        """
        Compute the bandwidth of the chirp signal.

        Returns:
            bw (float64): The bandwidth of the chirp signal in Hz. Format: float64.
            
        >>> test = task_2_2()
        >>> bw = test.get_bw_chirp()
        >>> (bw >= 100) & (bw <= 1000)
        True
        """
        s_t = self.chirp_data["values"] # signal values
        fs = self.chirp_data["fs"] # sampling frequency
        
        bw = 0
        
        # >>>>>>>>>>>>>>> YOUR CODE HERE <<<<<<<<<<<<<<<
        # TODO:
        # >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<
        s_f = fft(s_t)
        s_f = np.abs(s_f)
        s_f_freq = fftfreq(len(s_t), 1/fs)
        n = len(s_f)
        s_f = s_f[:n//2]
        s_f_freq = s_f_freq[:n//2]
        # plt.plot(s_f_freq, s_f)
        # plt.show()
        freqs, _ = find_peaks(s_f)
        freqs = s_f_freq[freqs]
        bw = freqs[-1] - freqs[0]
        return bw 
    
    def get_heart_rate(self):
        """
        Determine the heart rate from the ECG signal.

        Returns:
            hr (float64): The heart rate in beats per minute (BPM).
        
        >>> test = task_2_2()
        >>> hr = test.get_heart_rate()
        >>> (hr >= 60) & (hr <= 90)
        True
        """
        s_t = self.ecg_data["values"] # signal values
        fs = self.ecg_data["fs"] # sampling frequency
        
        hr = 0
        
        # >>>>>>>>>>>>>>> YOUR CODE HERE <<<<<<<<<<<<<<<
        # TODO:
        # >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<
        s_f = fft(s_t)
        s_f = np.abs(s_f)
        s_f_freq = fftfreq(len(s_t), 1/fs)
        n = len(s_f)
        s_f = s_f[:n//2]
        s_f_freq = s_f_freq[:n//2]
        # plt.plot(s_f_freq[int(0.5*fs):int(3*fs)], s_f[int(0.5*fs):int(3*fs)])
        # plt.show()
        freq, _ = find_peaks(s_f)
        peak_idx = freq[np.argmax(s_f[freq])]
        peak = s_f_freq[peak_idx]
        hr = peak * 60
        # Make sure hr is a float64
        if isinstance(hr, np.ndarray):
            if hr.size > 1:
                hr = hr[0]
            hr = hr.item()
        if isinstance(hr, list):
            if len(hr) > 1:
                hr = hr[0]
        hr = float(hr)
        return hr

if __name__ == "__main__":
    data_root = "./data/" # Change this to the directory where you store the data
    test = task_2_2(data_root=data_root)
    freq = test.get_heart_rate()
    print(freq)
    # ...