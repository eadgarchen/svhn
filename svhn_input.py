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

