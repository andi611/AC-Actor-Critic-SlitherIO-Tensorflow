import numpy as np
from random import shuffle, randint
#--------------------------------------------------------------#
import gym
from gym.spaces.box import Box
#--------------------------------------------------------------#
import universe
from universe import spaces
from universe import vectorized
from universe.wrappers import  EpisodeID, Vision, Logger
from universe.wrappers.gym_core import gym_core_action_space
#--------------------------------------------------------------#
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
universe.configure_logging()
#--------------------------------------------------------------#


class CropScreen(vectorized.ObservationWrapper):
	"""Crops out a [height]x[width] area starting from (top,left) """
	def __init__(self, env, height, width, top=0, left=0):
		super(CropScreen, self).__init__(env)
		self.height = height
		self.width = width
		self.top = top
		self.left = left
		self.observation_space = Box(0, 255, shape=(height, width, 3))
	def _observation(self, observation_n):
		return [ob[self.top:self.top+self.height, self.left:self.left+self.width, :] if ob is not None else None
				for ob in observation_n]


def create_internet_env(env_id, remotes, **_):
	env = gym.make(env_id)
	env = Vision(env) # extracts the vision modality and discards all others
	env = Logger(env)
	env = CropScreen(env, height=300, width=500, top=84, left=18) # Crops out a [height]x[width] area from screen starting from (top,left)
	env = EpisodeID(env)
	env.configure(remotes=remotes)
	return env



actions  = [universe.spaces.PointerEvent(270, 340, 0), # up
			universe.spaces.PointerEvent(270, 100, 0), # down
			universe.spaces.PointerEvent(150, 250, 0), # left
			universe.spaces.PointerEvent(400, 250, 0)] # right

def randomAgent():
	action = randint(0,3)
	if action == 0:
		return [universe.spaces.PointerEvent(270, 340, 0)]
	elif action == 1:
		return [universe.spaces.PointerEvent(270, 100, 0)]
	elif action == 2:
		return [universe.spaces.PointerEvent(150, 250, 0)]
	elif action == 3:
		return [universe.spaces.PointerEvent(400, 250, 0)]


env = create_internet_env(env_id='internet.SlitherIO-v0', remotes=1)
observation_n = env.reset() # returns a vector of observations of n environments

count = 0
while count <= 2000:
	#shuffle(actions)
	action_space = [randomAgent()]  # your agent here
	# submits a vector of actions; one for each environment instance it is controlling
	observation_n, reward_n, done_n, info = env.step(action_space)
	print('observation space:', env.observation_space.shape)
	print('observation:',np.shape(observation_n[0]))
	print('reward:', reward_n[0])
	env.render()
	count += 1
	if count == 2000:
		visualize = np.asarray(observation_n[0])
		print(visualize.shape)
		from PIL import Image
		img = Image.fromarray(visualize, 'RGB')
		img.save('visualize.png')
		img.show()

