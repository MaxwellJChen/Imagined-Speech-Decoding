#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import useful packages
import numpy as np
import pandas as pd
import scipy.io

def DatasetLoader(DIR):
    '''

    This is the Data Loader for our Library.
    The Dataset was supported via .csv file.
    In the CSV file, each line is a sample.
    For training or testing set, the columns are features of the EEG signals
    For training and testing labels, the columns are corresponding labels.
    In details, please refer to https://github.com/SuperBruceJia/EEG-Motor-Imagery-Classification-CNNs-TensorFlow
    to load the EEG Motor Movement Imagery Dataset, which is a benchmark for EEG Motor Imagery.

    Args:
        train_data: The training set for your Model
        train_labels: The corresponding training labels
        test_data: The testing set for your Model
        test_labels: The corresponding testing labels
        one_hot: One-hot representations for labels, if necessary

    Returns:
        train_data:   [N_train X M]
        train_labels: [N_train X 1]
        test_data:    [N_test X M]
        test_labels:  [N_test X 1]
        (N: number of samples, M: number of features)

    '''

    # Read Training Data and Labels
    train_data = pd.read_csv(DIR + 'training_set.csv', header=None)
    train_data = np.array(train_data).astype('float32')

    train_labels = pd.read_csv(DIR + 'training_label.csv', header=None)
    train_labels = np.array(train_labels).astype('float32')
    # If you met the below error:
    # ValueError: Cannot feed value of shape (1024, 1) for Tensor 'input/label:0', which has shape '(1024,)'
    # Then you have to uncomment the below code:
    # train_labels = np.squeeze(train_labels)
    
    # Read Testing Data and Labels
    test_data = pd.read_csv(DIR + 'test_set.csv', header=None)
    test_data = np.array(test_data).astype('float32')

    test_labels = pd.read_csv(DIR + 'test_label.csv', header=None)
    test_labels = np.array(test_labels).astype('float32')
    # If you met the below error:
    # ValueError: Cannot feed value of shape (1024, 1) for Tensor 'input/label:0', which has shape '(1024,)'
    # Then you have to uncomment the below code:
    # test_labels = np.squeeze(test_labels)

    return train_data, train_labels, test_data, test_labels

def printer(arr):
    print(f"Shape: {np.shape(arr)}")
    print(f"Max: {np.max(arr)}")
    print(f"Min: {np.min(arr)}")

def printer(arr):
    print(f"Shape: {np.shape(arr)}")
    print(f"Max: {np.max(arr)}")
    print(f"Min: {np.min(arr)}")

def nguyen_load():
    path = '/Users/maxwellchen/Desktop/Nguyen_et_al/Short_words/'
    files = ['sub_1_ch64_s_eog_removed_256Hz.mat', 'sub_3_ch64_s_eog_removed_256Hz.mat',
             'sub_5_ch64_s_eog_removed_256Hz.mat', 'sub_6b_ch80_s_eog_removed_256Hz.mat',
             'sub_8g_ch64_s_eog_removed_256Hz.mat', 'sub_12b_ch64_s_eog_removed_256Hz.mat', 'eeg_data_wrt_task_rep_no_eog_256Hz_last_beep']

    all = scipy.io.loadmat(path + files[0])[files[6]]
    for i in range(1, 6):
      sub = scipy.io.loadmat(path + files[i])[files[6]]
      all = np.concatenate((all, sub), axis = 1)
      print(np.shape(all))

    # DESTROY list format!
    out = np.expand_dims(np.concatenate([np.expand_dims(trial, axis = 0)[:, :64, :] for trial in all[0]], axis = 0), axis = 0)
    IN = np.expand_dims(np.concatenate([np.expand_dims(trial, axis = 0)[:, :64, :] for trial in all[1]], axis = 0), axis = 0)
    up = np.expand_dims(np.concatenate([np.expand_dims(trial, axis = 0)[:, :64, :] for trial in all[2]], axis = 0), axis = 0)

    # Remove rest
    all = np.concatenate((out, IN, up), axis = 0)
    printer(all)
    all = all[:, :, :, :768]

    # SPLIT into TRIALS!!!
    out = all[0]
    IN = all[1]
    up = all[2]

    out = np.concatenate(np.split(out, 3, axis=2), axis = 0)
    IN = np.concatenate(np.split(IN, 3, axis = 2), axis = 0)
    up = np.concatenate(np.split(up, 3, axis = 2), axis = 0)
    printer(out)
    printer(IN)
    printer(up)

    out = np.array(out)
    IN = np.array(IN)
    up = np.array(up)
    ranges = np.concatenate([range(30), range(300, 330), range(600, 630), range(900, 930), range(1200, 1230), range(1500, 1530)])
    printer(ranges)

    # Test set
    out_test = out[ranges][:][:]
    in_test = IN[ranges][:][:]
    up_test = up[ranges][:][:]
    x_test = np.concatenate((out_test, in_test, up_test), axis = 0)

    # Train set
    out_train = np.delete(out, ranges, axis = 0)
    in_train = np.delete(IN, ranges, axis = 0)
    up_train = np.delete(up, ranges, axis = 0)
    x_train = np.concatenate((out_train, in_train, up_train), axis = 0)

    # Test labels
    y_test = np.zeros((540, 3))
    y_test[:180, 0] = 1
    y_test[180:360, 1] = 1
    y_test[360:, 2] = 1

    # Train labels
    y_train = np.zeros((4860, 3))
    y_train[:1620, 0] = 1
    y_train[1620:3240, 1] = 1
    y_train[3240:, 2] = 1

    printer(x_train)
    printer(x_test)
    printer(y_train)
    printer(y_test)
    return x_train, y_train, x_test, y_test

def norm(epoch):
    final = []
    for x in range(len(epoch)):
        normalized = np.expand_dims((epoch[x] - np.min(epoch[x])) / (np.max(epoch[x]) - np.min(epoch[x])), axis=0)
        if x == 0:
            final = normalized
        else:
            final = np.concatenate((final, normalized), axis=0)
        if (x + 1) % 100 == 0:
            print(x + 1)
    return final

def norm_calc():
    x_train = np.load('x_train.npy')
    x_test = np.load('x_test.npy')
    x_train = norm(x_train)
    x_test = norm(x_test)

    np.save('x_train', x_train)
    np.save('x_test', x_test)

# def norm_load():
