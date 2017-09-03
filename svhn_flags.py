""" A deep conv network for implementing the Multi-digit Number Recognition on Street View House Number(SVHN).
This solution based on the paper from Goodfellow @ Google.
<Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks>
https://arxiv.org/pdf/1312.6082v4.pdf

Author: CHEN Wen
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os

"""
Directory Flags
"""
tf.app.flags.DEFINE_string('svhndata_path', '/home/wchen/codes/svhn/dataset/',
	"""Directory where the dataset located.""")
"""
Training and Inference Flags
"""
tf.app.flags.DEFINE_integer('image_size', 32,
	"""Image size, both height and length are same, image_size x image_size.""")
tf.app.flags.DEFINE_integer('batch_size', 64,
	"""Batch size.""")
tf.app.flags.DEFINE_integer('patch_size', 5,
	"""The conv kernel patch size.""")
tf.app.flags.DEFINE_integer('depth1', 16,
	"""The 1st conv layer depth.""")
tf.app.flags.DEFINE_integer('depth2', 32,
	"""The 2nd conv layer depth.""")
tf.app.flags.DEFINE_integer('depth3', 64,
	"""The 3rd conv layer depth.""")
tf.app.flags.DEFINE_integer('num_hidden1', 64,
	"""The number neurals in hidden 1 layer.""")
tf.app.flags.DEFINE_integer('num_hidden2', 32,
	"""The number neurals in hidden 2(full connectly) layer.""")
tf.app.flags.DEFINE_integer('num_channels', 1,
	"""The number of channels in the dataset image.""")
tf.app.flags.DEFINE_integer('num_labels', 10,
	"""The number of labels for the model output.""")
tf.app.flags.DEFINE_integer('training_datasize', 20000,
	"""The number of training data size.""")