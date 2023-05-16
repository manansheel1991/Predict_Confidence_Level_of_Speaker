# -*- coding: utf-8 -*-
"""
Created on Wed May 10 14:28:50 2023

@author: manan
"""
import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import wave
import webrtcvad
mysp=__import__("my-voice-analysis")
from scipy.io import wavfile
import contextlib
import io
import os
import re

#filename = 'common_voice_en (1).wav'
#temp_filename = 'common_voice_en (1)_temp.wav'

def extract_frequency_related_information(filename):
    
    ## Function to extract fundamental frequency, mean fundamental frequency, range of frequencies
    
    y, samplerate = sf.read('{}'.format(filename)) 
    y = y[:, 0]
    chunks = np.array_split(y,int(samplerate/2000))
    peaks = []
    
    for chunk in chunks:
        wavel = chunk
        # compute the magnitude of the Fourier Transform and its corresponding frequency values
        freq_magnitudes = np.abs(np.fft.rfft(wavel))
        freq_values = np.fft.rfftfreq(len(wavel), 1/samplerate)
        # find the max. magnitude
        max_positive_freq_idx = np.argmax(freq_magnitudes)
        peaks.append(freq_values[max_positive_freq_idx])
        
    ff = np.min(peaks)
    mean_ff = np.mean(peaks)
    range_ff = np.max(peaks) - np.min(peaks)
    
    return ff, mean_ff, range_ff

#ff, mean_ff, range_ff = extract_frequency_related_information(filename)

def extract_amplitude_related_information(filename):
    
    ## Function to extract mean amplitude, amplitude variance

    y, samplerate = sf.read('{}'.format(filename))
    
    y = y[:, 0]
    
    amplitude_mean = np.mean(abs(y))
    
    amplitude_variance = np.var(abs(y))
    
    return amplitude_mean, amplitude_variance

#a_m, a_v = extract_amplitude_related_information(filename)

#p = 'Sample_Audio_1'
#
#c = r'C:\Users\manan\Documents\Confidence Information Speech Task - Interview\Common-Voice-Files-In-WAV-Format'
#
#temp_filename = '{filename}_temp.wav'

def convert_audio_file(filename, temp_filename, path):
    
    #y, samplerate = sf.read('{}'.format(filename))
    
    y, samplerate = librosa.load(r'C:\Users\manan\Documents\Confidence Information Speech Task - Interview\Common-Voice-Files-In-WAV-Format\{}'.format(filename), sr = 44100)

    #y = librosa.core.resample(y, orig_sr = samplerate, target_sr = 44100)
    
    sf.write(r'{}/{}'.format(path, temp_filename), y, 44100, 'PCM_16')
    
    #librosa.output.write_wav(f"{path}/{temp_filename}", y, samplerate)
    

def analyze_audio_file(filename, temp_filename, path):
    
    ## Function to extract Speech Rate, Pause_Frequency, Balance, Articulation_rate
    
    convert_audio_file(filename, temp_filename, path)
    with io.StringIO() as buf, contextlib.redirect_stdout(buf):
        mysp.mysptotal(temp_filename[:-4], path)
        #mysp.mysptotal(filename[:-4], path)
        captured_output = buf.getvalue()
        
        numbers = [float(num) for num in re.findall(r"\d+\.\d+|\d+", captured_output) if num != "0"]
        numbers = numbers[:14]
        
        # remove temp file
        os.remove(fr"{path}/{temp_filename}")

        audio_params = {
            "number_of_syllables": numbers[0],
            "number_of_pauses": numbers[1],
            "rate_of_speech": numbers[2],
            "articulation_rate": numbers[3],
            "speaking_duration": numbers[4],
            "original_duration": numbers[5],
            "balance": numbers[6],
            "pause_frequency": numbers[1]/numbers[5]
            
        }
        
        return audio_params['rate_of_speech'], audio_params['pause_frequency'], audio_params['balance'], audio_params['articulation_rate']
      
#a, b, o, t = analyze_audio_file(filename, temp_filename, path = c)

# convert_audio_file(filename, temp_filename, c)


