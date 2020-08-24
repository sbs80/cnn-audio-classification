#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
"""Train Classifier from the UrbanSound8K dataset

This script takes extracted features from the UrbanSound8K dataset and trains a convolutional neural network for audio classification.

The script requires the extracted features to be found in the file 'features.pt' as a list of Pytorch tensors along with their associated audio class numbers and data fold numbers. This file should first be generated using the associated script 'feature_extract.py'

The user can define the fold number (1-10) that is used for validation using the -f flag. The best model according to its accuracy when applied to the validation fold is saved in the folder 'models' within the current directory. A log file is also generated in the folder 'logs' within the current directory. The log file logs the model's performance at each epoch.

For more details about the UrbanSound8K dataset see https://urbansounddataset.weebly.com/urbansound8k.html

This script requires a CUDA enabled Pytorch installation with as well as the 'math', 'random' and 'tqdm' packages.

usage: python train.py -f <validation fold number (1-10)>

"""
import sys
import getopt
import os

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import math
import random

# Construction of neural network:
model = nn.Sequential(
        nn.Conv2d(1, 24, 3),
        nn.ReLU(),
        nn.BatchNorm2d(24),
        nn.MaxPool2d(2),
        nn.Conv2d(24, 32, 3),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.MaxPool2d(2),
        nn.Conv2d(128, 256, 3),
        nn.ReLU(),
        nn.BatchNorm2d(256),
        nn.AdaptiveMaxPool2d(1),
        nn.Flatten(),
        nn.Linear(256,128),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(128,64),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(64,10),
        nn.Softmax(1)
        )
model.cuda()

# Construction of loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, amsgrad=True)

def process_batch(batch):
    """Function to process a batch of data with the current model and return loss and accuracy measurements for the batch.

    Args:
        batch: a batch of features to process.

    Returns:
        batch_loss: loss measurement for the model applied to the batch.
        batch_accuracy: accuracy measurement of the model applied to the batch.

    """
    features = []
    actual_classes = []

    for j in range(len(batch)):
        feature_tensor = batch[j][0]
        out_tensor = torch.tensor([batch[j][1]])

        with torch.no_grad():

            features.append(feature_tensor)
            actual_classes.append(out_tensor)

    features = torch.stack(features, 0).cuda()
    actual_classes = torch.stack(actual_classes).view(-1).cuda()

    # Process features through current model:
    outputs = model.forward(features)
       
    # Assign highest scoring class as predicted class for accuracy measurement
    predicted_classes = torch.argmax(outputs, 1)

    # Calculate accuracy of batch
    with torch.no_grad():
        total_count = len(batch)
        correct_prediction_count = (predicted_classes == actual_classes).sum()
        batch_accuracy = correct_prediction_count.item() / total_count

    # Calculate loss of batch
    batch_loss=criterion(outputs, actual_classes)

    return batch_loss, batch_accuracy


def main(argv):
    
    validation_fold = 10
    models_save_dir = "models"
    log_save_dir = "logs"
    features_file = "features.pt"

    batch_size = 100
    no_epochs = 200

    try:
        opts, args = getopt.getopt(argv, "hf:",["valfold="])
    except getopt.GetOptError:
        print("usage: python train.py -f <validition fold number (1-10)>")
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print("usage: python train.py -f <validation fold number (1-10)>")
            sys.exit()
        elif opt in ("-f", "--valfold"):
            try:
                validation_fold = int(arg)
            except ValueError:
                print("error: validation fold number must be an integer.")
                sys.exit(2)
            if validation_fold < 1 or validation_fold > 10:
                print("error: validation fold number must be an integer greater than 0 and less than 11.")
                sys.exit(2)
     
    print("Using fold " + str(validation_fold) + " as the validation fold.")

    if os.path.exists(models_save_dir) == False:
        os.mkdir(models_save_dir)
    elif os.path.isdir(models_save_dir) == False:
        print("error: '" + models_save_dir + "' should be a directory not a file.")
        sys.exit(1)

    if os.path.exists(log_save_dir) == False:
        os.mkdir(log_save_dir)
    elif os.path.isdir(log_save_dir) == False:
        print("error: '" + log_save_dir + "' should be a directory not a file.")
        sys.exit(1)

    log_file = open(os.path.join(log_save_dir, "log_fold_" + str(validation_fold) + ".csv"), "w+")
    log_file.write("epoch,train accuracy,train loss,validation accuracy,validation loss\n")

    # Load dataset:
    features = torch.load(features_file)

    train, val = [], []

    # Split dataset into training and validation sets:
    for f in features:
        if f[2] == validation_fold:
            val.append(f[:2])
        else:
            train.append(f[:2])

    del features

    best_accuracy = 0.0;
    best_epoch = 0;

    num_batches = math.ceil(len(train)/batch_size)

    for epoch in range(no_epochs):

        running_loss = 0.0
        running_accuracy = 0.0

        random.shuffle(train)

        model.train()

        print("training epoch " + str(epoch) + ":")

        for i in tqdm(range(0, len(train), batch_size)):

            optimizer.zero_grad()

            batch = train[i:i+batch_size]

            batch_loss, batch_accuracy = process_batch(batch)

            running_loss += batch_loss.item()
            running_accuracy += batch_accuracy

            batch_loss.backward()
            optimizer.step()
        
        print("no. of train samples: " + str(i))
        av_loss = running_loss/num_batches
        accuracy = 100*running_accuracy/num_batches
        print("train accuracy: " + "{:.2f}".format(accuracy) + "%")
        print("train loss: " + "{:.3f}".format(av_loss))

        log_file.write(str(epoch) + "," +  str(accuracy) + "," + str(av_loss) + ",")
        
        num_val_batches = math.ceil(len(val)/batch_size)

        running_loss = 0.0
        running_accuracy = 0.0

        model.eval()

        for i in range(0, len(val), batch_size):

            batch = val[i:i+batch_size]
            
            with torch.no_grad():
            
                batch_loss, batch_accuracy = process_batch(batch) 

            running_loss += batch_loss.item()
            running_accuracy += batch_accuracy

        av_loss = running_loss/num_val_batches
        accuracy = 100*running_accuracy/num_val_batches

        log_file.write(str(accuracy) + "," + str(av_loss) + "\n")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch
            print("new best validation accuracy! saving model..")

            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': av_loss
                }, os.path.join(models_save_dir,'best_model_validation_fold_' + str(validation_fold) + '.pt'))

        
        print("val accuracy: " + "{:.2f}".format(accuracy) + "%")
        print("val loss: " + "{:.3f}".format(av_loss))
        print("best val accuracy: " + "{:.2f}".format(best_accuracy) + "%")
        print("best epoch: " + str(best_epoch))
        print("most recent epoch: " + str(epoch))

    log_file.close()

if __name__ == "__main__":
    main(sys.argv[1:])
