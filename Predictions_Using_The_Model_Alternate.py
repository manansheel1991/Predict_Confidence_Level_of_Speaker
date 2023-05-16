# -*- coding: utf-8 -*-
"""
Created on Sun May 14 22:23:04 2023

@author: manan
"""
from Acoustic_Features_Extraction import extract_frequency_related_information, extract_amplitude_related_information, analyze_audio_file
from tensorflow.keras.models import load_model

model = load_model('Confidence_Scores_Model_Alternate.hdf5')

# Sample Audio File

filename = 'common_voice_en (32).wav'
filepath = r'C:\Users\manan\Documents\Predict_Confidence_Level_of_Speaker-main\Predict_Confidence_Level_of_Speaker-main\Common-Voice-Files-In-WAV-Format'

# Extract Features from the sample file

ff, mean_ff, range_ff = extract_frequency_related_information(r'{}/{}'.format(filepath, filename))
amplitude_mean, amplitude_variance = extract_amplitude_related_information(r'{}/{}'.format(filepath, filename))
#speech_rate, pause_frequency, balance, articulation_rate = analyze_audio_file(filename, temp_filename = '{}_temp.wav'.format(filename), path = filepath)
    
# Put features in an array for use by the model

xp = [[ff, mean_ff, range_ff, amplitude_mean, amplitude_variance]] #speech_rate, pause_frequency, balance, articulation_rate]] 

# Make a sample prediction

yp = model.predict(xp)

#print(yp)