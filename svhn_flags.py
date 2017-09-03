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

DATASET_PATH = "/home/wchen/codes/svhn/dataset/"

"""
Training and Inference Flags
"""

batch_size  = 64
patch_size = 5 # The conv kernel size
depth1 = 16
depth2 = 32
depth3 = 64
num_hidden1 = 64
num_hidden2 = 32
num_channels = 1 # Gray.
num_labels = 10

TRAINING_DATASIZE = 20000

