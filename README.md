# Predict Confidence Level of Speaker

## Introduction
The repository contains the code to extract acoustic features from any speech file taken as an input, and use these features to train a simple neural network model, that is able to predict the confidence level of the speaker.

## Acoustic Features taken to predict the confidence level of speech
1. Fundamental Frequency
2. Mean Fundamental Frequency
3. Range of frequencies
4. Amplitude Mean
5. Amplitude Variance
6. Rate of Speech
7. Pauses Frequency
8. Balance Ratio of Speech Duration and Original Duration
9. Articulation Rate

## Acoustic Feature Extraction
The file 'Acoustics_Feature_Extraction.py' contains the functions used for extracting the acoustic features. While the first 5 acoustics features are extracted using the knowledge of signal properties, the last 4 are extracted using a PRAAT script and a python package available online.

## Speech files used for training
A few files from the mozilla common-voice dataset were used for training the model. In reality thousands of files are required, but a few were used for demonstration purposes in the code. The features of all the files were extracted to create a list of features that would be the independent variables for training. As there is no publicly available dataset that has confidence scores associated with it, a random array was created as the code is for demonstration purposes. The training happens in the file 'Confidence_Model_Building.py'.






