""" A deep conv network for implementing the Multi-digit Number Recognition on Street View House Number(SVHN).

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
import svhn_input
import lcn

FLAGS = None

def inference(x, shape):
	"""
	Build a 7-layer CNN. Conv kernel size is 5 x 5. Conv stride size is 1 x 1.Valid padding.
	Max pool for sub-sampling, window size is 2 x 2, stride size is 2 x 2.Same padding.

	C1: Conv layer, Conv kernel size: 5 x 5 x 1 x 16, batch_size x 28 x 28 x 16.
	S2: Sub-sampling layer, Max pool window size is 2 x 2, batch_size x 14 x 14 x 16
	C3: Conv layer, Conv kerne size: 5 x 5 x 16 x 32, batch_size x 10 x 10 x 32
	S4: Sub-sampling layer, max pool window size is 2 x 2, batch_size x 5 x 5 x 32
	C5: Conv layer, Conv kernel size: 5 x 5 x 32 x 64, batch_size x 1 x 1 x 64
	Dropout
	F6: Fully connected layer, weight size: 64 x 16
	Output layer, weight size: 16 x 10



	Args:
		x: an input tensor with the dimensions (N_examples, 32 x 32 x 3)

	Returns:
		A tuple(y, keep_drop), y is a tensor of shape (N_examples, 10) with vaules
		equal to the logits of classifying the digit into one of 10 class(the digits 0 - 9).
		keep_drop is a scalar placeholder for the probability of the dropout.
	"""
	# keep_prob for all hidden layers.
	keep_prob = tf.placeholder(tf.float32, name = 'keep_drop')

	LCN = lcn.LecunLCN(x, shape)

	# Start build the convolutional layer and max pool layer.
	# The 1st convolutional layer - map 1 features (gray) image to 16 features.
	# Output of 1st is 14 x 14 x 16
	with tf.name_scope('conv1'):
		W_conv1 = weight_variable([svhn_flags.patch_size, svhn_flags.patch_size, svhn_flags.num_channels, svhn_flags.depth1])
		b_conv1 = bias_variable(svhn_flags.depth1)
		s_conv1 = stride_variable([1, 1, 1, 1])
		h_conv1 = tf.nn.relu(conv2d(LCN, W_conv1, s_conv1) + b_conv1)
	
	# The 1st local response normaliziation
	with tf.name_scope('lrn1'):
		h_lrn1 = tf.nn.lrn(h_conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

	# The 1st max pooling layer - downsamples by 2x2. Output is 14 x 14 x 16
	with tf.name_scope('pool1'):
		s_pool1 = stride_variable([1, 2, 2, 1])
		h_pool1 = max_pool_2x2(h_lrn1, s_pool1)

	# The 2nd convolutional layer - map 16 features to 32 features.
	# Output of 2nd is 5 x 5 x 32
	with tf.name_scope('conv2'):
		W_conv2 = weight_variable([svhn_flags.patch_size, svhn_flags.patch_size, svhn_flags.depth1, svhn_flags.depth2])
		b_conv2 = bias_variable(svhn_flags.depth2)
		s_conv2 = stride_variable([1, 1, 1, 1])
		h_conv2 = tf.nn.relu(conv2d(h_norm1, W_conv2, s_conv2) + b_conv2)

	# The 2nd local response normalization
	with tf.name_scope('lrn2'):
		h_lrn2 = tf.nn.lrn(h_conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

	# The 2nd max pooling layer - downsamples by 2x2
	with tf.name_scope('pool2'):
		s_pool2 = stride_variable([1, 2, 2, 1])
		h_pool2 = max_pool_2x2(h_lrn2, s_pool2)

	# The 3rd convolutional layer - map 32 features to 64 features.
	# Output of 3rd is 1 x 1 x 64.
	with tf.name_scope('conv3'):
		W_conv3 = weight_variable([svhn_flags.patch_size, svhn_flags.patch_size, svhn_flags.depth2, svhn_flags.depth3])
		b_conv3 = bias_variable(svhn_flags.depth3)
		s_conv3 = stride_variable([1, 1, 1, 1])
		h_conv3 = tf.nn.relu(conv2d(h_drop2, W_conv3, s_conv3) + b_conv3)

	# The dropout.
	with tf.name_scope('dropout'):
		h_drop = tf.nn.dropout(h_conv3, keep_prob)

	# Build the fully connected layer.
	# Map 1 x 1 x 64 to 64 x 32 features.
	with tf.name_scope('fc1'):
		W_fc1 = weight_variable([svhn_flags.num_hidden1, svhn_flags.num_hidden2])
		b_fc1 = bias_variable([svhn_flags.num_hidden2])

		shape = h_drop.get_shape().as_list()
		h_lc1_flat = tf.reshape(h_drop, [shape[0], shape[1] * shape[2] * shape[3]])
		h_fc1 = tf.nn.relu(tf.matmul(h_lc1_flat, W_fc1) + b_fc1)

	# Output.Map the [None, 32] features to [None, 10] classes, one for each digit
	with tf.name_scope('softmax'):
		W_fc2 = weight_variable([svhn_flags.num_hidden2, svhn_flags.num_labels])
		b_fc2 = bias_variable([svhn_flags.num_hidden2])
		y_conv = tf.matmul(h_drop5, W_fc2) + b_fc2
		
	return y_conv, keep_prob

def conv2d(x, W, strides):
	"""conv2d returns a 2d convolution layer with full stride."""
	return tf.nn.conv2d(x, W, strides, padding='VALID')

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
	# Import dataset, input_data is imported from the svhn_input.py
	# TODO...
	svhn_dataset = input_data.read_data_set()
	
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
	print('Saving graph to: %s' % graph_location)
	train_writer = tf.summary.FileWriter(graph_location)
	train_writer.add_graph(tf.get_default_graph())
	
	# Training and inference
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		# Training the graph
		for i in range(svhn_flags.TRAINING_DATASIZE):
			# NEED TO CHANGE With SVHN DATASET
			batch = svhn_dataset.train.next_batch(svhn_flags.BATCH_SIZE)
			if i % 100 == 0:
				train_accuracy = accuracy.eval(feed_dict = {
					x : batch[0], y_ : batch[1], keep_prob: 1.0})
				print('step %d, training accuracy %g' % (i, train_accuracy))
			train_step.run(feed_dict = {x: batch[0], y_ : batch[1], keep_prob : 1.0})

		# Test the accuracy of the inference
		print('test accuracy %g' % accuracy.eval(feed_dict = {
			x : svhn_dataset.test.images, y_ : svhn_dataset.test.labels, keep_prob : 1.0}))

	pass

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str,
                    default='/tmp/svhn/input_data',
                    help='Directory for storing input data')
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)