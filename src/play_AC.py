#--------------------------------------------------------------#
import universe
from universe.spaces import PointerEvent
#--------------------------------------------------------------#
import numpy as np
import tensorflow as tf
#--------------------------------------------------------------#
from AC import Actor, Critic
from env import createEnv
from utils import preprocess_screen, visualize
from utils import transform_acton, repeat_action, env_wrapper
#--------------------------------------------------------------#
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
universe.configure_logging()
#--------------------------------------------------------------#
import os
if not os.path.exists('_data'): os.makedirs('_data')
#--------------------------------------------------------------#


env = createEnv(env_id='internet.SlitherIO-v0', remotes=1)


#--------------------------------------------------------------#

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
actor = Actor(sess, n_features=[150, 250, 1], n_actions=8)
critic = Critic(sess, n_features=[150, 250, 1])
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver()
saver.restore(sess, '_data/model.ckpt') # load model

#--------------------------------------------------------------#

MAX_EPISODE = 500
PRINT = bool(0)
RENDER = bool(1)

#--------------------------------------------------------------#


env.reset()
READY = False
t = 0
while True:
	if RENDER: env.render()
	if not READY:
		a = 4
		action = [transform_acton(a)] # dummy action
		s, r, done = env_wrapper(env.step(action))
		# when observation is ready, start training
		if np.shape(s) == (300, 500, 3):
			screen = preprocess_screen(s)
			READY = True
	else:
		probs, a = actor.choose_action(screen, PLAY=True)
		action = [transform_acton(a)]
		s, r, done = env_wrapper(env.step(action))

		if done:
			t = 0
			READY = False
			screen = np.zeros((150, 250, 1))
		else:
			screen = preprocess_screen(s)

		# Get screen image
		if PRINT:
			print('---------------')
			print('Time step:', t)
			print('Action :', a)
			print('Reward:', r)
			if (t % 5 == 0):
				visualize(screen=screen, time=t)

#--------------------------------------------------------------#

