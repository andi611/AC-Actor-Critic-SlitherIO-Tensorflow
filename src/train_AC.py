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

MAX_EPISODE = 500
VISUALIZE = bool(0)
PRINT = bool(1)
RENDER = bool(0)
LOAD = bool(1)

env = createEnv(env_id='internet.SlitherIO-v0', remotes=1)

#--------------------------------------------------------------#

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
actor = Actor(sess, n_features=[150, 250, 1], n_actions=8)
critic = Critic(sess, n_features=[150, 250, 1])
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver()

if LOAD:
	saver.restore(sess, '_data/model.ckpt') # load model
	reward_history = np.loadtxt('_data/_reward_history.txt').tolist() # load history
	timestep_history = np.loadtxt('_data/_timestep_history.txt').tolist() # load history
	system_message = 'Loading and continue training on existing model.'
else: 
	reward_history = []
	timestep_history = []
	system_message = 'Training new model from scratch.'

#--------------------------------------------------------------#

for i_episode in range(MAX_EPISODE):
	env.reset()
	READY = False
	t = 0
	R_t = []
	track_a = np.arange((8), dtype=np.int) # records the most recent 8 action
	penalty = -0.25

	while True:
		if RENDER: env.render()
		if not READY:
			a = 4
			v_, td_error, a_loss = 0, 0, 0
			action = [transform_acton(a)] # dummy action
			s, r, done = env_wrapper(env.step(action))
			# when observation is ready, start training
			if np.shape(s) == (300, 500, 3):
				screen = preprocess_screen(s)
				READY = True
		else:
			probs, a_ = actor.choose_action(screen, PLAY=False) # Set PLAY to False for training
			action = [transform_acton(a_)]
			s_, r, done = env_wrapper(env.step(action))
			if done: r = -10.0
			
			# reward shaping
			REPEAT, track_a, penalty = repeat_action(a_, track_a, penalty)
			if REPEAT: r += penalty
			if r > 0: r *= 2
			if a_ != a: r += 1
			R_t.append(r)

			# Get screen image
			if PRINT:
				print('---------------')
				print('Mode:', system_message)
				print('Time step:', t)
				print('History:', track_a)
				print('Action :', a_)
				print('Reward:', r)
				print('V_:', v_)
				print('TD:', td_error)
				print('Actor Loss:', a_loss)
				print('Action prob:', probs.ravel())
				if (t % 5 == 0 and VISUALIZE):
					visualize(screen=screen, time=t)

			if done:
				screen_ = np.zeros((150, 250, 1)) # a black screen for terminal state
				critic.learn(s=screen, r=r, s_=screen_)
				actor.learn(s=screen, a=a, td=td_error)
				ep_Rt_sum = sum(R_t)
				reward_history.append(ep_Rt_sum)
				timestep_history.append(t)
				print('--------------------------------------------------')
				print('episode:', i_episode)  
				print('reward:', int(ep_Rt_sum))
				print('--------------------------------------------------')
				break

			screen_ = preprocess_screen(s_)
			v_, td_error = critic.learn(s=screen, r=r, s_=screen_)  # gradient = grad[r + gamma * V(s_) - V(s)]
			a_loss = actor.learn(s=screen, a=a, td=td_error)     # true_gradient = grad[logPi(s,a) * td_error]
			
			screen = screen_
			a = a_
			t += 1

	save_path = saver.save(sess, '_data/model.ckpt')
	saver.restore(sess, '_data/model.ckpt') # reload model
	np.savetxt('_data/_reward_history.txt', reward_history)
	np.savetxt('_data/_timestep_history.txt', timestep_history)
	print('----------------------------------------------------------')
	print('Model saved in file:', save_path)
	print('Reward history saved in file: _data/_reward_history.txt')
	print('Timestep history saved in file: _data/_timestep_history.txt')
	print('----------------------------------------------------------')

#--------------------------------------------------------------#


