# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ env.py ]
#   Synopsis     [ environment wrapper for Slither.Io ]
#   Author       [ Ting-Wei Liu (Andi611) ]
#   Copyright    [ Copyleft(c), NTUEE, NTU, Taiwan ]
"""*********************************************************************************************"""

###############
# IMPORTATION #
###############
import gym
from gym.spaces.box import Box
from universe import vectorized
from universe.wrappers import EpisodeID, Vision, Logger


##############
# CROPSCREEN #
##############
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


##############
# CREATEENV #
##############
def createEnv(env_id, remotes, **_):
	env = gym.make(env_id)
	env = Vision(env) # extracts the vision modality and discards all others
	env = Logger(env)
	env = CropScreen(env, height=300, width=500, top=84, left=18) # Crops out a [height]x[width] area from screen starting from (top,left)
	env = EpisodeID(env)
	env.configure(remotes=remotes)
	return env


def main():
	env = createEnv(env_id='internet.SlitherIO-v0', remotes=1)


if __name__ == '__main__':
	main()