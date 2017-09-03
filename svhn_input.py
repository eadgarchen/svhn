""" A deep conv network for implementing the Multi-digit Number Recognition on Street View House Number(SVHN).
This solution based on the paper from Goodfellow @ Google.
<Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks>
https://arxiv.org/pdf/1312.6082v4.pdf

This program handles the test/train data from SVHN dataset,
currently focus on MNIST-like 32-by-32 format2 dataset.

Author: CHEN Wen
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import svhn_flags

from scipy.io import loadmat as load

FLAGS = tf.app.flags.FLAGS

svhndata_path = FLAGS.svhndata_path

def input_data():
	train_dataset = load(svhndata_path + 'train_32x32.mat', variable_names = 'X').get('X')
	train_labels = load(svhndata_path + 'train_32x32.mat', variable_names = 'y').get('y')
	test_dataset = load(svhndata_path + 'test_32x32.mat', variable_names = 'X').get('X')
	test_labels = load(svhndata_path + 'test_32x32.mat', variable_names = 'y').get('y')
	valid_dataset = load(svhndata_path + 'extra_32x32.mat', variable_names = 'X').get('X')
	valid_labels = load(svhndata_path + 'extra_32x32.mat', variable_names = 'y').get('y')
	
	print(train_dataset.shape, train_labels.shape)
	print(test_dataset.shape, test_labels.shape)
	print(valid_dataset.shape, valid_labels.shape)

	return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels