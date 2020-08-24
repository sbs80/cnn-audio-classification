#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
"""Feature Extractor

This script takes the original UrbanSound8K dataset and extracts features from the audio in the form of Mel spectrograms. The path to the dataset location can be defined by the user using the -d flag.

The sample-rates of the audio files are first converted to 16 kHz before Mel spectrograms are calculated.

The extracted features are saved in the current directory to the file 'features.pt' as a list of Pytorch tensors along with their associated audio class numbers and data fold numbers.

For more details about the UrbanSound8K dataset see https://urbansounddataset.weebly.com/urbansound8k.html

This script requires a Pytorch installation as well as the 'torchaudio', 'pandas', 'tqdm' packages.

usage: python feature_extract.py -d <path_to_dataset>

"""
import sys
import os
import getopt

import torch
import torch.nn as nn
import torchaudio
import pandas as pd
from tqdm import tqdm

# Number of samples of audio to use:
window_size = 64000

def main(argv):

    # Default path to dataset:
    data_path = 'UrbanSound8K'
    # Filename of saved features:
    features_file_name = 'features.pt'

    # Parse command line arguments:
    try:
        opts, args = getopt.getopt(argv,"hd:",["datapath="])
    except getopt.GetoptError:
        print('usage: python feature_extract.py -d <path_to_dataset>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('usage: python feature_extract.py -d <path_to_dataset>')
            sys.exit()
        elif opt in ("-d", "--datapath"):
            data_path = arg
    
    print('Data path is: ', data_path)

    # Try to read the metadata file from the UrbanSound8K data directory:
    try:
        data = pd.read_csv(os.path.join(data_path, "metadata", "UrbanSound8K.csv"))
    except:
        print("Error: couldn't read metadata file, check data path.")
        sys.exit(1)
    
    features = []

    audio_path = os.path.join(data_path, "audio")

    # Use the torchaudio Sox interface to reduce number of channels to 1 and convert the sample rate to 16 kHz:
    E = torchaudio.sox_effects.SoxEffectsChain(normalization=True)
    E.append_effect_to_chain("remix", "1")
    E.append_effect_to_chain("rate", 16000)

    bad_data_count = 0

    # Iterate through audio files and extract features:
    for i in tqdm(range(len(data))):
        fold_no = data.iloc[i]["fold"]
        filename = data.iloc[i]["slice_file_name"]
        label = data.iloc[i]["classID"]
        filepath = os.path.join(audio_path, "fold" + str(fold_no), filename)
        
        E.set_input_file(filepath)
        file_tensor, sr = E.sox_build_flow_effects()

        if file_tensor.size()[1] < window_size:
            pad_amount = window_size - file_tensor.size()[1]
            file_tensor = nn.functional.pad(file_tensor,(0,pad_amount))
        elif file_tensor.size()[1] > window_size:
            file_tensor = file_tensor[:,0:window_size]

        file_features = torchaudio.transforms.MelSpectrogram(n_fft=2048,hop_length=512)(file_tensor)
        file_features = torchaudio.transforms.AmplitudeToDB()(file_features)
        
        if(file_features.isnan().any()):
            bad_data_count = bad_data_count + 1
        else:
            features.append([file_features, label, fold_no])    

    try:
        torch.save(features, features_file_name)
    except:
        print("Error: couldn't save features to file.")
        sys.exit(1)

    print("Feature extraction successful!")
    print("Features saved in the file " + features_file_name)
    
    if bad_data_count > 0:
        print("Note: " + str(bad_data_count) + " files were rejected possibly owing to corrupt data.")

if __name__ == "__main__":
    main(sys.argv[1:])
