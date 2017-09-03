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

FLAGS = tf.app.flags.FLAGS

image_size = FLAGS.image_size
batch_size  = FLAGS.batch_size
patch_size = FLAGS.patch_size
depth1 = FLAGS.depth1
depth2 = FLAGS.depth2
depth3 = FLAGS.depth3
num_hidden1 = FLAGS.num_hidden1
num_hidden2 = FLAGS.num_hidden2
num_channels = FLAGS.num_channels
num_labels = FLAGS.num_labels
training_datasize = FLAGS.training_datasize

shape = [batch_size, image_size, image_size, num_channels]

train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = svhn_input.input_data()

def accuracy(predictions, labels):
	return (100.0 * np.sum(np.argmax(predictions, 1) == labels)
		/ predictions.shape[0])

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

def model(x, keep_prob, shape):
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
		keep_prob: keep prob
		shape: shape

	Returns:
		A tuple(y, keep_drop), y is a tensor of shape (N_examples, 10) with vaules
		equal to the logits of classifying the digit into one of 10 class(the digits 0 - 9).
		keep_drop is a scalar placeholder for the probability of the dropout.
	"""
	LCN = lcn.LecunLCN(x, shape)

	# Start build the convolutional layer and max pool layer.
	# The 1st convolutional layer - map 1 features (gray) image to 16 features.
	# Output of 1st is 14 x 14 x 16
	with tf.name_scope('conv1'):
		W_conv1 = weight_variable([patch_size, patch_size, num_channels, depth1])
		b_conv1 = bias_variable(depth1)
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
		W_conv2 = weight_variable([patch_size, patch_size, depth1, depth2])
		b_conv2 = bias_variable(depth2)
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
		W_conv3 = weight_variable([patch_size, patch_size, depth2, depth3])
		b_conv3 = bias_variable(depth3)
		s_conv3 = stride_variable([1, 1, 1, 1])
		h_conv3 = tf.nn.relu(conv2d(h_drop2, W_conv3, s_conv3) + b_conv3)

	# The dropout.
	with tf.name_scope('dropout'):
		h_drop = tf.nn.dropout(h_conv3, keep_prob)

	# Build the fully connected layer.
	# Map 1 x 1 x 64 to 64 x 32 features.
	with tf.name_scope('fc1'):
		W_fc1 = weight_variable([num_hidden1, num_hidden2])
		b_fc1 = bias_variable([num_hidden2])

		shape = h_drop.get_shape().as_list()
		h_lc1_flat = tf.reshape(h_drop, [shape[0], shape[1] * shape[2] * shape[3]])
		h_fc1 = tf.nn.relu(tf.matmul(h_lc1_flat, W_fc1) + b_fc1)

	# Output.Map the [None, 32] features to [None, 10] classes, one for each digit
	with tf.name_scope('softmax'):
		W_fc2 = weight_variable([num_hidden2, num_labels])
		b_fc2 = bias_variable([num_hidden2])
		y_conv = tf.matmul(h_drop5, W_fc2) + b_fc2
		
	return y_conv

def main(unused_argv):
	"""
	"""
	# Create the x, y placeholder.
	tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
	tf_train_labels = tf.placeholder(tf.int64, shape=(batch_size))
	tf_valid_dataset = tf.constant(valid_dataset)
	tf_test_dataset = tf.constant(test_dataset)

	# Create the model
	y_conv, keep_prob = model(tf_train_dataset, 0.9735, shape)
	
	# Define the loss
	with tf.name_scope('loss'):
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = tf_train_labels,
																logits = y_conv)
	loss = tf.reduce_mean(cross_entropy)
	
	with tf.name_scope('adam_optimizer'):
		optimizer = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	
	# Predictions for the training, validation and test data.
	train_prediction = tf.nn.softmax(model(tf_train_dataset, 1.0, shape))
	valid_prediction = tf.nn.softmax(model(tf_valid_dataset, 1.0, shape))
	test_prediction = tf.nn.softmax(model(tf_test_dataset, 1.0, shape))

	# Save the graph
	graph_location = tempfile.mkdtemp()
	print('Saving graph to: %s' % graph_location)
	train_writer = tf.summary.FileWriter(graph_location)
	train_writer.add_graph(tf.get_default_graph())
	
	# Training and inference
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		# Training the graph
		for step in range(training_datasize):
			# Get the batch data.
			offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
			batch_data = train_dataset[offset : (offset + batch_size), :, :, :]
			batch_labels = train_labels[offset : (offset + batch_size)]

			feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
			_, l, predictions = session.run(
				[optimizer, loss, train_prediction], feed_dict = feed_dict)

			if step % 500 == 0:
				print('Minibatch loss at step %d: %f' % (step, l))
				print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))

				# Inference and test the accuracy of the model on the validate dataset every 500 steps.
				print('Validation accuracy: %.1f%%' % accuracy(valid_prediction, valid_labels))

		# Inference and test the accuracy of the model on test dataset.
		print('Test accuracy : %.1f%%' % accuracy(test_prediction.eval(), test_labels))

		# TODO: Save the model...

if __name__ == '__main__':
	tf.app.run()