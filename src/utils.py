#--------------------------------------------------------------#
from universe.spaces import PointerEvent
from PIL import Image
import numpy as np
#--------------------------------------------------------------#

def transform_acton(action):
	assert 0 <= action <= 7
	if action == 0:
		return [PointerEvent(270, 340, 0)] # up
	elif action == 1:
		return [PointerEvent(270, 100, 0)] # down
	elif action == 2:
		return [PointerEvent(150, 250, 0)] # left
	elif action == 3:
		return [PointerEvent(400, 250, 0)] # right
	elif action == 4:
		return [PointerEvent(150, 340, 0)] # up left
	elif action == 5:
		return [PointerEvent(400, 340, 0)] # up right
	elif action == 6:
		return [PointerEvent(150, 100, 0)] # bottum left
	elif action == 7:
		return [PointerEvent(400, 100, 0)] # bottum right


def preprocess_screen(screen):
	# resize image from (300, 500, 3) to (150, 250, 3), scaling ratio = 0.5
	im = Image.fromarray(screen, mode='RGB')
	im = im.resize(size=(250, 150), resample=Image.BILINEAR) # size = (width, height)
	# turn (150, 250, 3) RGB to (150, 250, 1) grayscale
	im = np.asarray(im)
	im = (0.299*im[:,:,0] + 0.587*im[:,:,1] + 0.114*im[:,:,2]).reshape((150, 250, 1))
	# returns a normalized screen
	im /= 255
	return im


def shift(array):
	array_ = np.empty_like(array)
	array_[:-1] = array[1:]
	array_[-1] = -8
	return array_


def repeat_action(action, history, penalty):
	# record
	history = shift(history)
	history[-1] = action
	# check redundent
	if history[-1] == history[-2] == history[-3]:
		penalty -= 0.25 # increment penalty
		return True, history, penalty
	else:
		penalty = -0.25 # reset penalty
		return False, history, penalty


def visualize(screen, time):
	screen *= 255
	name = '_data/slither_' + str(time) + '.png'
	im = Image.fromarray(screen.squeeze())
	im.convert('RGB').save(name)


def env_wrapper(env_step):
	s = env_step[0][0]
	r = env_step[1][0]
	done = env_step[2][0]
	info = env_step[3]
	return s, r, done

