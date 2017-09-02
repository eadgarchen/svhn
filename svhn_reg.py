""" A deep conv network for implementing the Multi-digit Number Recognition on Street View House Number(SVHN).
This solution based on the paper from Goodfellow @ Google.
<Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks>
https://arxiv.org/pdf/1312.6082v4.pdf

Author: CHEN Wen
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

import tensorflow as tf
import svhn_flags

FLAGS = None

def inference(x):
	"""
	Build the nerual network:8 convolutional hidden layer, 1 locally connected hidden layer,
	2 densely connected hidden layers.
	The number of units at each spatial location in each layer is [48, 64, 128, 160] 
	for the first 4 nerual layers,
	192 for all other locally connected layers,	and 3072 for all fully connected layers.


	X: 128x128x3 input image.
	Conv Kernel Size: all conv kernel size is 5x5.
	Max Pool Window Size: all max pool window size is 2x2.
	Stride: the stride alternates between 2 and 1 at each layer.
	Padding: zero padding.
	Each convolutional layer includes max pooling and subtractive normalization.
	Dropout: we trained with dropout applied to all hidden layers but not the input.

	Args:
		x: an input tensor with the dimensions (N_examples, 128x128x3)

	Returns:
		A tuple(y, keep_drop), y is a tensor of shape (N_examples, 10) with vaules
		equal to the logits of classifying the digit into one of 10 class(the digits 0 - 9).
		keep_drop is a scalar placeholder for the probability of the dropout.
	"""
	# keep_prob for all hidden layers.
	keep_prob = tf.placeholder(tf.float32, name = 'keep_drop')

	# Reshape...
	with tf.name_scope('reshape'):
		x_image = tf.reshape(x, [-1, 128, 128, 3])

	# Start build the convolutional layer and max pool layer.
	# The 1st convolutional layer - map 3 features (RGB) image to 48 features.
	# Output of 1st is 64x64x48
	with tf.name_scope('conv1'):
		W_conv1 = weight_variable([5, 5, 3, 48])
		b_conv1 = bias_variable(48)
		s_conv1 = stride_variable([1, 1, 1, 1])
		h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, s_conv1) + b_conv1)

	# The 1st max pooling layer - downsamples by 2x2
	with tf.name_scope('pool1'):
		s_pool1 = stride_variable([1, 2, 2, 1])
		h_pool1 = max_pool_2x2(h_conv1, s_pool1)

	# The 1st subtractive normalization
	with tf.name_scope('norm1'):
		h_norm1 = tf.nn.lrn(h_pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

	# The 1st dropout.
	with tf.name_scope('drop1'):
		h_drop1 = tf.nn.dropout(h_norm1, keep_prob)

	# The 2nd convolutional layer - map 48 features to 64 features.
	# Output of 2nd is 64x64x64, max pool window size is 1x1.
	with tf.name_scope('conv2'):
		W_conv2 = weight_variable([5, 5, 48, 64])
		b_conv2 = bias_variable(64)
		s_conv2 = stride_variable([1, 1, 1, 1])
		h_conv2 = tf.nn.relu(conv2d(h_norm1, W_conv2, s_conv2) + b_conv2)

	# The 2nd max pooling layer - downsamples by 2x2
	with tf.name_scope('pool2'):
		s_pool2 = stride_variable([1, 1, 1, 1])
		h_pool2 = max_pool_2x2(h_conv2, s_pool2)

	# The 2nd subtractive normalization
	with tf.name_scope('norm2'):
		h_norm2 = tf.nn.lrn(h_pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

	# The 2nd dropout.
	with tf.name_scope('drop2'):
		h_drop2 = tf.nn.dropout(h_norm2, keep_prob)

	# The 3rd convolutional layer - map 64 features to 128 features.
	# Output of 3rd is 32x32x128.
	with tf.name_scope('conv3'):
		W_conv3 = weight_variable([5, 5, 64, 128])
		b_conv3 = bias_variable(128)
		s_conv3 = stride_variable([1, 1, 1, 1])
		h_conv3 = tf.nn.relu(conv2d(h_drop2, W_conv3, s_conv3) + b_conv3)

	# The 3rd max pooling layer - downsamples by 2x2
	with tf.name_scope('pool3'):
		s_pool3 = stride_variable([1, 2, 2, 1])
		h_pool3 = max_pool_2x2(h_conv3, s_pool3)

	# The 3rd subtractive normalization
	with tf.name_scope('norm3'):
		h_norm3 = tf.nn.lrn(h_pool3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

	# The 3rd dropout.
	with tf.name_scope('drop3'):
		h_drop3 = tf.nn.dropout(h_norm3, keep_prob)

	# The 4th convolutional layer - map 128 features to 160 features.
	# Output of 4th is 32x32x160
	with tf.name_scope('conv4'):
		W_conv4 = weight_variable([5, 5, 128, 160])
		b_conv4 = bias_variable(160)
		s_conv4 = stride_variable([1, 1, 1, 1])
		h_conv4 = tf.nn.relu(conv2d(h_drop3, W_conv4, s_conv4) + b_conv4)

	# The 4th max pooling layer - downsamples by 2x2
	with tf.name_scope('pool4'):
		s_pool4 = stride_variable([1, 1, 1, 1])
		h_pool4 = max_pool_2x2(h_conv4, s_pool4)

	# The 4th subtractive normalization
	with tf.name_scope('norm4'):
		h_norm4 = tf.nn.lrn(h_pool4, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

	# The 4th dropout.
	with tf.name_scope('drop4'):
		h_drop4 = tf.nn.dropout(h_norm4, keep_prob)

	# Build the locally connected layer.
	# After 4 round of downsampling, our 128x128x3 image is down to
	# 32x32x160 feature maps(see each pooling layer), now map this to 192 features.
	with tf.name_scope('lc1'):
		W_lc1 = weight_variable([8 * 8 * 160, 192])
		b_lc1 = bias_variable([192])
	
		h_drop4_flat = tf.reshape(h_drop4, [-1, 8 * 8 * 160])
		h_lc1 = tf.nn.relu(tf.matmul(h_drop4_flat, W_lc1) + b_lc1)

	# The locally connected layer dropout.
	with tf.name_scope('drop5'):
		h_drop5 = tf.nn.dropout(h_lc1, keep_prob)

	# Build the fully connected layer.
	# Map 32x32x192 to 32x32x3072 features.
	with tf.name_scope('fc1'):
		W_fc1 = weight_variable([192, 3072])
		b_fc1 = bias_variable([3072])
	
		h_lc1_flat = tf.reshape(h_lc1, [-1, 192])
		h_fc1 = tf.nn.relu(tf.matmul(h_lc1_flat, W_fc1) + b_fc1)

	# The fully connected layer dropout.
	with tf.name_scope('drop6'):
		h_drop5 = tf.nn.dropout(h_fc1, keep_prob)

	# Map the 32x32x3072 features to [None, 10] classes, one for each digit
	with tf.name_scope('softmax'):
		W_fc2 = weight_variable([3072, 10])
		b_fc2 = bias_variable([10])

		y_conv = tf.nn.xw_plus_b(h_drop5, W_fc2, b_fc2)

	
	return y_conv, keep_prob

def conv2d(x, W, strides):
	"""conv2d returns a 2d convolution layer with full stride."""
	return tf.nn.conv2d(x, W, strides, padding='SAME')

def max_pool_2x2(x, strides):
	"""max_pool_2x2 downsamples a feature map by 2X."""
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides = strides, padding='SAME')

def weight_variable(shape):
	"""weight_variable generates a weight variable of a given shape."""
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	"""bias_variable generates a bias variable of a given shape."""
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def stride_variable(shape):
	"""stride_variable generates a stide variable of a given shape."""
	initial = tf.constant(1, shape=shape)
	return tf.Variable(initial)

def main():
	"""
	"""
	# Import data from svhn_input, input_data is imported from the svhn_input.py
	# TODO...
	svhn = input_data.read_data_set()
	
	# Create the x, y_
	x = tf.placeholder(tf.float32, [None, 784])
	y_ = tf.placeholder(tf.float32, [None, 10])

	# Create the model
	y_conv, keep_prob = inference(x)
	
	# Define the loss
	with tf.name_scope('loss'):
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = y_,
																logits = y_conv)
	cross_entropy = tf.reduce_mean(cross_entropy)
	
	with tf.name_scope('adam_optimizer'):
		train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	
	# Define the accuracy
	with tf.name_scope('accuracy'):
		correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
		correct_prediction = tf.cast(correct_prediction, tf.float32)
	accuracy = tf.reduce_mean(correct_prediction)

	# Save the graph
	graph_location = tempfile.mkdtemp()
	print('Saving graph to: %s', % graph_location)
	train_writer = tf.summary.FileWriter(graph_location)
	train_writer.add_graph(tf.get_default_graph())
	
	# Training and inference
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		# Training the graph
		for i in range(svhn_flags.TRAINING_DATASIZE):
			# NEED TO CHANGE With SVHN DATASET
			batch = svhn.train.next_batch(svhn_flags.BATCH_SIZE)
			if i % 100 == 0:
				train_accuracy = accuracy.eval(feed_dict = {
					x : batch[0], y_ : batch[1], keep_prob: 1.0})
					})
				print('step %d, training accuracy %g' % (i, train_accuracy))
			train_step.run(feed_dict = {x: batch[0], y_ : batch[1], keep_prob : 1.0})

		# Test the accuracy of the inference
		print('test accuracy %g' % accuracy.eval(feed_dict = {
			x : svhn.test.images, y_ : svhn.test.labels, keep_prob : 1.0}
			}))

	pass

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str,
                    default='/tmp/tensorflow/mnist/input_data',
                    help='Directory for storing input data')
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)