import numpy as np
import matplotlib.pyplot as plt

reward_history = np.loadtxt('_data/_reward_history.txt') # load history
timestep_history = np.loadtxt('_data/_timestep_history.txt') # load history
print('Number of trained episode:')
print(len(reward_history))
print(len(timestep_history))

print('Number of timesteps:')
print(len(reward_history)*np.sum(timestep_history))
x = np.arange(len(reward_history))

print('Average reward', np.mean(reward_history))
print('Average timestep', np.mean(timestep_history))

#plt.plot(x, reward_history) # blue
#plt.plot(x, timestep_history, 'r') # red
reward_per_step = reward_history/timestep_history
for i in range(len(reward_per_step)):
	if reward_per_step[i] > 3: reward_per_step[i] = 2.5
plt.plot(x, reward_per_step, 'g') # green

plt.xlabel("Episodes")
#plt.ylabel("Episodic Reward")
#plt.ylabel("Episodic Timestep")
#plt.ylabel("Episodic Reward / Timestep")
plt.ylabel("Episodic Reward per Timestep")
#plt.ylabel("Episodic normalized Reward")
#plt.title('Episodic Reward v.s. Episode')
#plt.title('Timestep v.s. Episode')
#plt.title('Reward + Timestep v.s. Episode')

plt.show()
