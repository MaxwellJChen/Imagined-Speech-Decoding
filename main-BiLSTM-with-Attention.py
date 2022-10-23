#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import useful packages
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

# Hide the Configuration and Warnings
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

import random
import numpy as np
import tensorflow as tf
from DataLoader import printer
from BiLSTM_with_Attention import BiLSTM_with_Attention
from Loss import loss
from Metrics import evaluation

# Model Name
Model = 'Attention_based_Long_Short_Term_Memory'

# Clear all the stack and use GPU resources as much as possible
tf.compat.v1.reset_default_graph()
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

# Your Dataset Location, for example EEG-Motor-Movement-Imagery-Dataset
# The CSV file should be named as training_set.csv, training_label.csv, test_set.csv, and test_label.csv
# DIR = 'DatasetAPI/EEG-Motor-Movement-Imagery-Dataset/'
SAVE = '/Users/maxwellchen/Desktop/Deis/Results/BiLSTM-with-Attention/'
if not os.path.exists(SAVE):  # If the SAVE folder doesn't exist, create one
    os.mkdir(SAVE)

# Load the dataset, here it uses one-hot representation for labels
train_data = np.load('/Users/maxwellchen/PycharmProjects/A_BiLSTM/x_ica1_train.npy')
train_labels = np.load('/Users/maxwellchen/PycharmProjects/A_BiLSTM/y_ica1_train.npy')
test_data = np.load('/Users/maxwellchen/PycharmProjects/A_BiLSTM/x_ica1_test.npy')
test_labels = np.load('/Users/maxwellchen/PycharmProjects/A_BiLSTM/y_ica1_test.npy')
print('\n' + '\n' +'\n' + '\n' + '42') # PRINT HERE
printer(train_data)
printer(train_labels)
printer(test_data)
printer(test_labels)

print('\n' + '\n' +'\n' + '\n' + '48') # PRINT HERE
train_labels = tf.one_hot(indices=train_labels, depth=3)
train_labels = tf.compat.v1.squeeze(train_labels).eval(session=sess)
test_labels = tf.one_hot(indices=test_labels, depth=3)
test_labels = tf.squeeze(test_labels).eval(session=sess)

# Model Hyper-parameters
n_input   = 60      # The input size of signals at each time
max_time  = 256      # The unfolded time slices of the BiLSTM Model
lstm_size = 256     # The number of LSTMs inside the BiLSTM Model
attention_size = 8  # The number of neurons of fully-connected layer inside the Attention Mechanism

n_class   = 3     # The number of classification classes
n_hidden  = 64    # The number of hidden units in the first fully-connected layer
num_epoch = 300   # The number of Epochs that the Model run
keep_rate = 0.75  # Keep rate of the Dropout

lr = tf.constant(1e-4, dtype=tf.float32)  # Learning rate
lr_decay_epoch = 50    # Every (50) epochs, the learning rate decays
lr_decay       = 0.50  # Learning rate Decay by (50%)

batch_size = 1024
n_batch = train_data.shape[0] // batch_size

# Initialize Model Parameters (Network Weights and Biases)
# This Model only uses Two fully-connected layers, and u sure can add extra layers DIY
print('\n' + '\n' +'\n' + '\n' + '74') # PRINT HERE
weights_1 = tf.Variable(tf.truncated_normal([2 * lstm_size, n_hidden], stddev=0.01))
biases_1  = tf.Variable(tf.constant(0.01, shape=[n_hidden]))
weights_2 = tf.Variable(tf.truncated_normal([n_hidden, n_class], stddev=0.01))
biases_2  = tf.Variable(tf.constant(0.01, shape=[n_class]))

# Define Placeholders
print('\n' + '\n' +'\n' + '\n' + '81') # PRINT HERE
x = tf.placeholder(tf.float32, [None, 60, 256])
y = tf.placeholder(tf.float32, [None, 3])
keep_prob = tf.placeholder(tf.float32)

# Load Model Network
print('\n' + '\n' +'\n' + '\n' + '87') # PRINT HERE
prediction, features, attention_weights = BiLSTM_with_Attention(Input=x,
                                                                max_time=max_time,
                                                                lstm_size=lstm_size,
                                                                n_input=n_input,
                                                                attention_size=attention_size,
                                                                keep_prob=keep_prob,
                                                                weights_1=weights_1,
                                                                biases_1=biases_1,
                                                                weights_2=weights_2,
                                                                biases_2=biases_2)

# Load Loss Function
# Load Optimizer
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

# Load Evaluation Metricse
Global_Average_Accuracy = evaluation(y=y, prediction=prediction)

# Merge all the summaries
print('\n' + '\n' +'\n' + '\n' + '109') # PRINT HERE
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(SAVE + '/train_Writer', sess.graph)
test_writer = tf.summary.FileWriter(SAVE + '/test_Writer')

# Initialize all the variables
print('\n' + '\n' +'\n' + '\n' + '115') # PRINT HERE
sess.run(tf.global_variables_initializer())

print('\n' + '\n' +'\n' + '\n' + '117') # PRINT HERE
for epoch in range(num_epoch + 1):
    # U can use learning rate decay or not
    # Here, we set a minimum learning rate
    # If u don't want this, u definitely can modify the following lines
    # print('\n' + '\n' + '\n' + '\n' + '1')  # PRINT HERE
    learning_rate = sess.run(lr)
    if epoch % lr_decay_epoch == 0 and epoch != 0:
        if learning_rate <= 1e-6:
            lr = lr * 1.0
            sess.run(lr)
        else:
            lr = lr * lr_decay
            sess.run(lr)
    # print('\n' + '\n' + '\n' + '\n' + '2')  # PRINT HERE
    # Randomly shuffle the training dataset and train the Model
    for batch_index in range(n_batch):
        random_batch = random.sample(range(train_data.shape[0]), batch_size)
        batch_xs = train_data[random_batch]
        batch_ys = train_labels[random_batch]
        sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: keep_rate})

    # print('\n' + '\n' + '\n' + '\n' + '3')  # PRINT HERE
    # Show Accuracy and Loss on Training and Test Set
    # Here, for training set, we only show the result of first 100 samples
    # If u want to show the result on the entire training set, please modify it.
    train_accuracy, train_loss = sess.run([Global_Average_Accuracy, loss], feed_dict={x: train_data[:100], y: train_labels[:100], keep_prob: 1.0})
    Test_summary, test_accuracy, test_loss = sess.run([merged, Global_Average_Accuracy, loss], feed_dict={x: test_data, y: test_labels, keep_prob: 1.0})
    test_writer.add_summary(Test_summary, epoch)
    # print('hi')
    # Show the Model Capability
    print("Iter " + str(epoch) + ", Testing Accuracy: " + str(test_accuracy) + ", Training Accuracy: " + str(train_accuracy))
    print("Iter " + str(epoch) + ", Testing Loss: " + str(test_loss) + ", Training Loss: " + str(train_loss))
    print("Learning rate is ", learning_rate)
    print('\n')

    # Save the prediction and labels for testing set
    # The "labels_for_test.csv" is the same as the "test_label.csv"
    # We will use the files to draw ROC CCurve and AUC
    if epoch == num_epoch:
        output_prediction = sess.run(prediction, feed_dict={x: test_data, y: test_labels, keep_prob: 1.0})
        np.savetxt(SAVE + "prediction_for_test.csv", output_prediction, delimiter=",")
        np.savetxt(SAVE + "labels_for_test.csv", test_labels, delimiter=",")

    # if you want to extract and save the features from fully-connected layer, use all the dataset and uncomment this.
    # All data is the total data = training data + testing data
    # We use the features from the overall dataset
    # ML models might be used to classify the features further
    # if epoch == num_epoch:
    #     Features = sess.run(Features, feed_dict={x: all_data, y: all_labels, keep_prob: 1.0})
    #     np.savetxt(SAVE + "Features.csv", Features, delimiter=",")

    # # U can output the Attention Weights at the final output of the Model, and visualize it.
    # Uncomment it, and save the Attention Weights
    # if epoch == num_epoch:
    #     attention_weights = sess.run(attention_weights, feed_dict={x: all_data, y: all_labels, keep_prob: 1.0})
    #     np.savetxt(SAVE + "attention_weights.csv", attention_weights, delimiter=",")

train_writer.close()
test_writer.close()
sess.close()
