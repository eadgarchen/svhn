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
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import svhn_flags

from scipy.io import loadmat as load

FLAGS = tf.app.flags.FLAGS

svhndata_path = FLAGS.svhndata_path


def img2gray(iamge):
	'''Normalize images.
	Apply the algorithm in this paper:http://www.eyemaginary.com/Rendering/TurnColorsGray.pdf
	'''
	image = image.astpye(float)
	image_gray = np.dot(image, [0.2989], [0.5870], [0.1140])

	return image_gray

def GCN(image):
	'''Global Contrast Normalization
	TODO: What's the purpose of this codes?
	'''

	imgsize = image.shape[0]
	mean = np.mean(image, axis = (1, 2), dtype = float)
	std = np.std(image, axis = (1, 2), dtype = float, ddof = 1)
	std[std < 1e-4] = 1.
	image_GCN = np.zeros(image.shape, dtype = float)

	for i in np.arange(imgsize):
		image_GCN[i, :, :] = (image[i, :, :] -mean[i]) / std[i]

	return image_GCN

def input_data():
	train_dataset = load(svhndata_path + 'train_32x32.mat', variable_names = 'X').get('X')
	train_labels = load(svhndata_path + 'train_32x32.mat', variable_names = 'y').get('y')
	test_dataset = load(svhndata_path + 'test_32x32.mat', variable_names = 'X').get('X')
	test_labels = load(svhndata_path + 'test_32x32.mat', variable_names = 'y').get('y')
	valid_dataset = load(svhndata_path + 'extra_32x32.mat', variable_names = 'X').get('X')
	valid_labels = load(svhndata_path + 'extra_32x32.mat', variable_names = 'y').get('y')
	
	n_labels = 10
	valid_index1 = []
	valid_index2 = []
	train_index1 = []
	train_index2 = []

	random.seed()

	for i in np.arange(n_labels):
		valid_index1.extend(np.where(train_labels[:, 0] == (i))[0][ : 400].tolist())
		train_index2.extend(np.where(train_labels[:, 0] == (i))[0][400 : ].tolist())
		valid_index2.extend(np.where(train_labels[:, 0] == (i))[0][ : 200].tolist())
		train_index2.extend(np.where(train_labels[:, 0] == (i))[0][200 : ].tolist())
	
	random.shuffle(valid_index1)
	random.shuffle(train_index1)
	random.shuffle(valid_index2)
	random.shuffle(train_index2)

	valid_dataset = np.concatenate((valid_dataset[:,:,:,valid_index2], train_dataset[:,:,:,valid_index1]), axis=3).transpose((3,0,1,2))
	valid_labels = np.concatenate((valid_labels[valid_index2,:], train_labels[valid_index1,:]), axis=0)[:,0]
	train_dataset = np.concatenate((valid_dataset[:,:,:,train_index2], train_dataset[:,:,:,train_index1]), axis=3).transpose((3,0,1,2))
	train_labels = np.concatenate((valid_labels[train_index2,:], train_labels[train_index1,:]), axis=0)[:,0]
	test_dataset = test_dataset.transpose((3,0,1,2))
	test_labels = test_labels[:,0]

	print(train_dataset.shape, train_labels.shape)
	print(test_dataset.shape, test_labels.shape)
	print(valid_dataset.shape, valid_labels.shape)

	return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels