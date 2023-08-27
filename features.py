import os
import random 
from pathlib import Path

import torchaudio
import numpy as np
from scipy import signal
import pandas as pd
import torch
from plot_audio import plot_mel_specgram

ALLOWED_CLASSES = ['normal', 'murmur', 'extrahls', 'artifact']

class Audio_Features:
    """
        This class prepare the features to be used in the CNN with the audio signals from Dataset
    """

    def __init__(self, output_path, data_path):
        """
            Initializes feature extraction routine

            Parameters
            ----------
                output_path: str
                to save the processed data
                data_path  Path:
                    Path with the raw data
        """

        self._output_path = Path(output_path)
        self._raw_data = Path(data_path)
        self._set = data_path.split("/")[-1]
        self._metadata_path = Path("./Data/metadata.csv")
        #random_filenames = self._get_filenames()
        self._save_features()

    def _downsampling(self, waveform, sample_rate):
        """
            Initializes feature extraction routine

            Parameters
            ----------
                output_path: str
                    Path to save the processed data
                data_path:
                    Path with the raw data
            Retunrs
            ----------
                wavefomr: np.array
                    Resamplled waveform 44100 Hz -> 4410 Hz
        """
        downsampling_scale = 10
        resample_frequency = int(sample_rate / downsampling_scale)

        resampled_waveform = torchaudio.functional.resample(waveform, 
                                                            sample_rate,
                                                            resample_frequency,
                                                            resampling_method="sinc_interp_kaiser")

        return resampled_waveform, resample_frequency
    
    def _reshape_audio(self, waveform, file_name):
        """
            Initializes feature extraction routine

            Parameters
            ----------
                output_path: str
                    Path to save the processed data
                data_path:
                    Path with the raw data
            Retunrs
            ----------
                wavefomr: np.array
                    Resamplled waveform 44100 Hz -> 4410 Hz
        """
        dataset_audio_size = self._metadata["num_frames"].min()
        audio_n_frames = self._metadata['num_frames'][self._metadata.index == file_name].values

        init_cut = abs(dataset_audio_size - audio_n_frames) # To prevent rizz of touch and make file lenght equal
        
        return waveform[:,init_cut[0]:]
    
    def _butterworth(self, waveform, sample_rate):
        """
            Initializes feature extraction routine

            Parameters
            ----------
                output_path: str
                    Path to save the processed data
                data_path:
                    Path with the raw data
            Retunrs
            ----------
                wavefomr: np.array
                    Resamplled waveform 44100 Hz -> 4410 Hz
        """
        filter_order = 3
        low_pass_freq = 195
        nyq = 0.5 * sample_rate
        normalized_low_freq = low_pass_freq / nyq

        sizes = waveform.size()
        waveform = waveform.reshape(sizes[-1]).numpy()

        z, p, k = signal.butter(filter_order, normalized_low_freq, btype='low', analog=False, output='zpk')
        b, a = signal.zpk2tf(z, p, k)
        zi = signal.lfilter_zi(b, a)

        filter_signal, _ = signal.lfilter(b, a, waveform, zi=zi*waveform[0])
        filter_signal = np.array(filter_signal, dtype='f')
        filter_signal = torch.from_numpy(filter_signal)
        filter_signal = filter_signal.reshape((1,sizes[-1]))

        return filter_signal

    def _mel_specgram(self, waveform, sample_rate, save_path):
        
        n_fft = 1024
        win_length = None
        hop_length = 512
        n_mels = 128

        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm='slaney',
            onesided=True,
            n_mels=n_mels,
            mel_scale="htk",
        )

        mfcc_image = mel_spec(waveform)
        plot_mel_specgram(mfcc_image[0], file_path=save_path)
       
        
    def _save_features(self):

        train_proportion = 0.85
        n_files = os.listdir(self._raw_data)
        train_test = np.random.choice([0, 1], size=len(n_files), p=[1-train_proportion, train_proportion])
        
        self._metadata = pd.read_csv(self._metadata_path,index_col=[0])
        
        for idx, file in enumerate(self._raw_data.glob("*")):

            class_type = str(file.name).split('_')[0]
            split_path = 'train' if train_test[idx]==1 else 'test'
            class_path = self._output_path / f"{self._set}_{split_path}" / class_type

            output_file_path = class_path / f"{str(file.name).split('.')[0]}.png"

            if  class_type in ALLOWED_CLASSES and file.name in self._metadata.index:
                if not os.path.exists(class_path):
                    os.makedirs(class_path)
            

                waveform, sample_rate = torchaudio.load(file)

                waveform = self._reshape_audio(waveform, file.name)
                waveform, resampled_rate = self._downsampling(waveform, sample_rate)
                waveform = self._butterworth(waveform, resampled_rate)
                
                self._mel_specgram(waveform, sample_rate=resampled_rate, save_path=output_file_path)
                print(output_file_path)
           
if __name__ == '__main__': 

    wav_path = './Data/set_b'

    Audio_Features(output_path='./Data', data_path=wav_path)
