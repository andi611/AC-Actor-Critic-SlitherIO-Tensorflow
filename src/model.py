#--------------------------------------------------------------#
import tensorflow as tf
#--------------------------------------------------------------#

"""""""""""""""
model structure:
Conv-32x7x7-s2
Conv-64x7x7-s2
MP-3x3-s2
Conv-128x3x3-s1
MP-3x3-s2
Conv-192x3x3-s1
FC-1024
"""""""""""""""

def CNN(inputs, n_features):
	# modify the n_feature list
	n_features = [-1] + n_features # [-1, 150, 250, x]
	# Input layer
	input_layer = tf.reshape(inputs, n_features)
	# Convolutional Layer #1
	conv1 = tf.layers.conv2d(
				inputs=input_layer,
				filters=32,
				kernel_size=[7, 7],
				strides=(2, 2),
				padding="same",
				activation=tf.nn.relu6,
				kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
				name='conv1')
	# Convolutional Layer #2
	conv2 = tf.layers.conv2d(
				inputs=conv1,
				filters=64,
				kernel_size=[7, 7],
				strides=(2, 2),
				padding="same",
				activation=tf.nn.relu6,
				kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
				name='conv2')
	# Pooling Layer #1
	pool1 = tf.layers.max_pooling2d(
				inputs=conv2,
				pool_size=[3, 3], 
				strides=2, 
				name='pool1')
	# Convolutional Layer #3
	conv3 = tf.layers.conv2d(
				inputs=pool1,
				filters=128,
				kernel_size=[3, 3],
				strides=(1, 1),
				padding="same",
				activation=tf.nn.relu6,
				kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
				name='conv3')
	# Pooling Layer #2
	pool2 = tf.layers.max_pooling2d(
				inputs=conv3, 
				pool_size=[3, 3], 
				strides=2, 
				name='pool2')
	# Convolutional Layer #3
	conv4 = tf.layers.conv2d(
				inputs=pool2,
				filters=192,
				kernel_size=[3, 3],
				strides=(1, 1),
				padding="same",
				activation=tf.nn.relu6,
				kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
				name='conv4')
	# flatten
	conv4_flat = tf.contrib.layers.flatten(conv4)
	# Dense Layer
	dense = tf.layers.dense(
				inputs=conv4_flat,
				units=1024,
				activation=tf.nn.relu6,
				kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
				name='dense')
	return dense

	