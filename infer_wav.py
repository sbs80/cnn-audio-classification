#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
"""Infer Wav File

This script uses a trained model to classify a single wav file according to the categories defined by the UrbanSound8K dataset. Currently the model used defaults to the model trained using folds 1-9 of the dataset. This is the default model that is trained and saved if the associated training script 'train.py' script is run successfully. Note that before this script can be run, the feature extraction script 'feature_extract' and the training script 'train.py' must first be run to train the model.

For more details about the UrbanSound8K dataset see https://urbansounddataset.weebly.com/urbansound8k.html

This script requires a CUDA enabled Pytorch installation as well as the 'torchaudio', 'pandas', 'tqdm' packages.

usage: python infer_wav.py -f <wav_file_name>

"""

import sys
import getopt
import os

import torch
import torch.nn as nn
import torchaudio
import pandas as pd
from tqdm import tqdm

from train import model
from feature_extract import window_size

classes = ['air conditioner', 'car horn', 'children playing', 'dog bark', 'drilling', 'engine idling', 'gun shot', 'jackhammer', 'siren', 'street music']

def main(argv):

    file_path = ''

    # Parse command line arguments:
    try:
        opts, args = getopt.getopt(argv,"hf:",["wavfile="])
    except getopt.GetoptError:
        print('usage: python infer_wav.py -f <wav_file_name>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('usage: python infer_wav.py -f <wav_file_name>')
            sys.exit()
        elif opt in ("-f", "--wavfile"):
            file_path = arg
    
    if not file_path:
        print('No audio file specified.')
        print('usage: python infer_wav.py -f <wav_file_name>')
        sys.exit(2)
    
    print('File to infer is: ', file_path)
 
    # Load model parameters:
    checkpoint = torch.load('models/best_model_validation_fold_10.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Use the torchaudio Sox interface to reduce number of channels to 1 and convert the sample rate to 16 kHz:
    E = torchaudio.sox_effects.SoxEffectsChain(normalization=True)
    E.append_effect_to_chain("remix", "1")
    E.append_effect_to_chain("rate", 16000)

    E.set_input_file(file_path)
    file_tensor, sr = E.sox_build_flow_effects()

    if file_tensor.size()[1] < window_size:
        pad_amount = window_size - file_tensor.size()[1]
        file_tensor = nn.functional.pad(file_tensor,(0,pad_amount))
    elif file_tensor.size()[1] > window_size:
        file_tensor = file_tensor[:,0:window_size]

    file_features = torchaudio.transforms.MelSpectrogram(n_fft=2048,hop_length=512)(file_tensor)
    file_features = torchaudio.transforms.AmplitudeToDB()(file_features)
        
    if(file_features.isnan().any()):
       print('Error processing audio.')
       sys.exit(1)
    
    #file_features.unsqueeze_(0)

    file_features = file_features.unsqueeze(0).cuda()

    with torch.no_grad():
        outputs = model.forward(file_features)


 
    # Assign highest scoring class as predicted class for accuracy measurement
    predicted_class = torch.argmax(outputs, 1)

    #print(outputs[0])

    print('Predicted class: ' + classes[predicted_class[0]])



if __name__ == "__main__":
    main(sys.argv[1:])

