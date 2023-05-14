# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 15:27:22 2019

@author: manan
"""
# Import necessary libraries

import pydub 
import scipy
import numpy as np
from pydub import AudioSegment
from scipy import signal
import librosa
from librosa import core
import matplotlib.pyplot as plt
from scipy.io import wavfile

from Acoustic_Features_Extraction import extract_frequency_related_information, extract_amplitude_related_information, analyze_audio_file

import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense

import random

import pandas as pd

# Initialize lists for acoustic features, 30 files are taken for the purpose of code demonstration.
# In reality, there should be thousands of files

ff = [None]*30
mean_ff = [None]*30
range_ff = [None]*30

amplitude_mean = [None]*30
amplitude_variance = [None]*30

speech_rate = [None]*30
pause_frequency = [None]*30
balance = [None]*30
articulation_rate = [None]*30

# For-loops to read the common-voice files, and extract acoustic features

for i in range(0,30):
    ff[i], mean_ff[i], range_ff[i] = extract_frequency_related_information(r'C:\Users\manan\Documents\Confidence Information Speech Task - Interview\Common-Voice-Files-In-WAV-Format\common_voice_en ({}).wav'.format(i+1))
    amplitude_mean[i], amplitude_variance[i] = extract_amplitude_related_information(r'C:\Users\manan\Documents\Confidence Information Speech Task - Interview\Common-Voice-Files-In-WAV-Format\common_voice_en ({}).wav'.format(i+1))

for i in range(0, 30):
    filename = 'common_voice_en ({}).wav'.format(i+1)
    speech_rate[i], pause_frequency[i], balance[i], articulation_rate[i] = analyze_audio_file(filename, temp_filename = '{}_temp.wav'.format(filename), path = r'C:\Users\manan\Documents\Confidence Information Speech Task - Interview\Common-Voice-Files-In-WAV-Format')
    
# Initialize Input Array of Independent Features

x = [[0 for i in range(9)] for j in range(30)] 

for i in range(0, 30):
        x[i][0] = ff[i]
        x[i][1] = mean_ff[i]
        x[i][2] = range_ff[i]
        x[i][3] = amplitude_mean[i]
        x[i][4] = amplitude_variance[i]
        x[i][5] = speech_rate[i]
        x[i][6] = pause_frequency[i]
        x[i][7] = balance[i]
        x[i][8] = articulation_rate[i]
        
# Define Target Confidence Scores for training the model, between 0 and 10. 
# As there is no dataset available with confidence scores, currently, this is put for code demonstration. 

y = [random.randint(0, 10) for _ in range(30)]

## Model in Keras

#define the model 
model = Sequential()

#add the layers
model.add(Dense(units=6, activation='relu', input_dim=9))
model.add(Dense(units=1))

#compile the model
model.compile(optimizer='sgd', loss='mean_squared_error')

#train the model
model.fit(x,y,epochs=100)

#Save the model to a file

model.save('Confidence_Scores_Model.hdf5', overwrite=True, include_optimizer=True)

# Make a Sample Prediction through the model

model = load_model('Confidence_Scores_Model.hdf5')

xp = [[250, 350, 400, 0.05, 0.006, 5, 0.56, 0.7, 5.0]]

yp = model.predict(xp)

