import numpy as np
import os.path as osp
import pickle
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq

# Radar parameters
c = 3e8           # Speed of light (m/s)
fc = 1e9          # Carrier frequency (Hz) 
B = 1.5e9         # Bandwidth (Hz)
T = 100e-6        # Chirp duration (s)
Fs = 2e6          # Sampling rate (Hz)
NUM_ANTENNAS = 4  # Number of antennas

class task_2_4:
    def __init__(self, data_root="./data/") -> None:
        """
        Initializes the task_2_4 class, loading various signal data from pickle files.

        Attributes:
            data_root (str): The root directory where data files are stored.
            rx_fn (str): Filename for the received signal (task_2_4).
        """
        self.data_root = data_root
        self.rx_fn = "task_2_4.pickle"
        
        
        with open(osp.join(self.data_root, self.rx_fn), "rb") as f:
            self.rx_data = pickle.load(f)
        
        self.num_samples = self.rx_data.shape[1]
        
        
    def generate_transmitted_signal(self):
        r"""
        Generate the transmitted signal based on the received signal.
        
        The chirp signal is defined as:
        \[
            s(t) = \exp\left(j \cdot 2\pi \cdot (f_s \cdot + \dfrac{B}{2 \cdot T} \cdot t)\cdot t\right)
        \]
        
        Returns:
            tx (np.ndarray): The transmitted signal.
        
        >>> task = task_2_4()
        >>> tx = task.generate_transmitted_signal()
        >>> round(tx[-1].imag, 1)
        -0.7
        """
        tx = None
        # >>>>>>>>>>>>>>> YOUR CODE HERE <<<<<<<<<<<<<<<
        # TODO:
        # >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<
        t = np.arange(self.num_samples) / Fs
        s_t = np.exp(1j * 2 * np.pi * (fc + B/(2*T) * t) * t)
        tx = s_t
        return tx
    
    def compute_if_signal(self):
        r"""
        Compute the IF signal based on the received signal.
        
        if_signal is given by:
        \[
            if_signal = s(t) \cdot r^*(t)
        \]
        
        Returns:
            if_signal (np.ndarray): The IF signal.
        
        >>> task = task_2_4()
        >>> if_signal = task.compute_if_signal()
        >>> round(if_signal[-1][-1].imag, 1)
        -1.3
        """
        tx = self.generate_transmitted_signal()
        if_signal = None
        # >>>>>>>>>>>>>>> YOUR CODE HERE <<<<<<<<<<<<<<<
        # TODO:
        # >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<
        tx = np.tile(tx, (NUM_ANTENNAS, 1))
        if_signal = tx * self.rx_data.conj()
        return if_signal
    
    def estimate_distance(self):
        """
        Estimate the distance based on the IF signal. In this case, there are two targets.
        
        Returns:
            distances (np.ndarray): The estimated distances (m) to the two targets in ascending order.
            range_fft (np.ndarray): The range FFT.
            range_bins (np.ndarray): The range bins corresponding to the range FFT (in meters).
        
        >>> task = task_2_4()
        >>> distances, _, _ = task.estimate_distance()
        >>> len(distances) == 2
        True
        """
        if_signal = self.compute_if_signal()
        distances = []
        range_fft = [] # Range FFT
        range_bins = None # Range bins
        # >>>>>>>>>>>>>>> YOUR CODE HERE <<<<<<<<<<<<<<<
        # TODO:
        # >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<
        # fig = plt.figure()
        for channel in if_signal:
            fft_if_signal = fft(channel)
            fft_if_signal_freq = fftfreq(len(channel), 1/Fs)[:len(channel)//2]
            if range_bins is None:
                range_bins = (c * fft_if_signal_freq * T) / (2 * B)
            fft_if_signal = np.abs(fft_if_signal)[:len(channel)//2]
            range_fft.append(fft_if_signal)
            peaks, _ = find_peaks(fft_if_signal, height=20)
            fd = range_bins[peaks]
            # distance = (c * fd * T) / (2 * B)
            distance = fd
            distances.extend(distance)
            # range_fft = fft_if_signal[peaks]
            # range_bins = fft_if_signal_freq[peaks]
        # plt.show()
        distances = np.unique(distances)
        distances = np.sort(distances)
        range_fft = np.array(range_fft).mean(axis=0)
        return distances, range_fft, range_bins
    
    def estimate_AoA(self):
        """
        Estimate the angle of arrival based on the received signal.
        
        Returns:
            aoas (dict): A dictionary containing the estimated AoA for each target. You should keep one decimal place for the angles.
        
        >>> task = task_2_4()
        >>> aoas = task.estimate_AoA()
        >>> len(aoas) == 2, type(aoas) == dict
        (True, True)
        >>> all(isinstance(v, float) for v in aoas.values())
        True
        """
        fd, range_fft, range_bins = self.estimate_distance()
        aoas = {}
        # >>>>>>>>>>>>>>> YOUR CODE HERE <<<<<<<<<<<<<<<
        # TODO:
        # >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<
        for target in fd:
            phase_diff = 2 * np.pi * target * T
            aoa = np.arcsin(phase_diff / (2 * np.pi * Fs * T))
            aoas[target] = np.degrees(aoa.mean())
        return aoas
    
if __name__ == "__main__":
    task =  task_2_4()
    # tx = task.generate_transmitted_signal()
    # print(tx)
    # task.compute_if_signal()
    # dist, rf, rb = task.estimate_distance()
    # print(dist)
    # print(rf)
    # print(rb)
    aoas = task.estimate_AoA()
    print(aoas)