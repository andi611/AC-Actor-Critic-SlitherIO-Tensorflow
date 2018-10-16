#--------------------------------------------------------------#
import tensorflow as tf
import numpy as np
from model import CNN
from random import Random
rand = Random()
# rand.seed(1337)
# np.random.seed(1337)
#--------------------------------------------------------------#

EPSILON = 0.2 # constant epsilon policy to encourage exploration
EXPLORATION_FACTOR = 0.2 # # multiply the policy output of the network by an exploration factor (0.2) before softmax
ENTROPY_BETA = 0.7 # discount factor for entropy
ENTROPY_CONSTANT = 1e-10 # to avoid taking log on zero probabilities, 1e-10

#--------------------------------------------------------------#

class Actor(object):
	def __init__(self, sess, n_features, n_actions, lr=0.001):
		self.sess = sess

		self.s = tf.placeholder(tf.float32, n_features, "state")
		self.a = tf.placeholder(tf.int32, None, "act")
		self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error
		with tf.variable_scope('Actor'):
			
			with tf.variable_scope('Actor_net'):
				# CNN network
				cnn = CNN(inputs=self.s, n_features=n_features)
				cnn_explore = cnn * EXPLORATION_FACTOR
				# Output layer
				self.acts_prob = tf.layers.dense(
					inputs=cnn_explore,
					units=n_actions,    # output units
					activation=tf.nn.softmax,   # get action probabilities
					kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
					name='acts_prob')

			with tf.variable_scope('a_loss'):
				log_prob = tf.log(self.acts_prob[0, self.a]) # log(Pi(s,a))
				exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss: log(Pi(s,a)) * td
				entropy = -tf.reduce_sum(self.acts_prob * tf.log(self.acts_prob + ENTROPY_CONSTANT)) # entropy is small when uniform
				self.a_loss = -tf.reduce_sum((exp_v + ENTROPY_BETA * entropy)) 

			with tf.name_scope('grad'):
				self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
				self.a_grads = tf.gradients(self.a_loss, self.a_params)

		with tf.name_scope('train'):
				self.train_op = tf.train.AdamOptimizer(lr).apply_gradients(zip(self.a_grads, self.a_params))

	def learn(self, s, a, td):
		feed_dict = {self.s: s, self.a: a, self.td_error: td}
		_, a_loss = self.sess.run([self.train_op, self.a_loss], feed_dict)
		return a_loss

	def choose_action(self, s, PLAY=False):
		probs = self.sess.run(self.acts_prob, {self.s: s})   # get probabilities for all actions
		if (rand.random() < (1 - EPSILON)) or PLAY:
			return probs, np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   # return a int
		else: return probs, rand.randint(0,7)

class Critic(object):
	def __init__(self, sess, n_features, lr=0.001):
		self.sess = sess

		self.s = tf.placeholder(tf.float32, n_features, "state")
		self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
		self.r = tf.placeholder(tf.float32, None, 'r')
		
		with tf.variable_scope('Critic'):

			with tf.variable_scope('Critic_Net'):
				# Convolutional Neutral network
				cnn = CNN(inputs=self.s, n_features=n_features)
				# Output layer
				self.v = tf.layers.dense(
					inputs=cnn,
					units=1,    # output units
					activation=None,   # linear
					kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
					name='V')

			with tf.variable_scope('squared_TD_error'):
				GAMMA = 0.9     # reward discount in TD error
				self.td_error = (self.r + GAMMA * self.v_) - self.v  # TD_error = (r+gamma*V_next) - V_curr
				self.c_loss = tf.square(self.td_error)

			with tf.name_scope('grad'):
				self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
				self.c_grads = tf.gradients(self.c_loss, self.c_params) 

		with tf.name_scope('train'):
			self.train_op = tf.train.AdamOptimizer(lr).apply_gradients(zip(self.c_grads, self.c_params))

	def learn(self, s, r, s_):
		v_ = self.sess.run(self.v, {self.s: s_})
		td_error, _ = self.sess.run([self.td_error, self.train_op],
										  {self.s: s, self.v_: v_, self.r: r})
		return v_, td_error

