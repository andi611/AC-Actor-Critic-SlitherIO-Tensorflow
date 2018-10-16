# Reinforcement Learning: Vision Based Agent trained with Actor Critic playing Slither.IO
**Training a vision-based agent with the Actor Critic model in an online environment, implementation in Tensorflow.**
[![Demo Video](https://github.com/andi611/RL_actor_critic_slitherio_Tensorflow/blob/master/data/gameplay.png)](https://youtu.be/8iRD1w73fDo)


## Requirements: 
* **Tensorflow**
* **Universe**
* Python


## Introduction
* In this report, we present the result of training a vision-based agent for Slither.io, an online massively multiplayer browser game that is partially supported by Universe (OpenAi), using Reinforcement Learning (RL) algorithms.
* The framework we used is based on the Actor-Critic models, combining with convolutional neural networks (CNN). 
* During training, we apply several techniques to encourage exploration and keeping our agent at a high entropy state, successfully avoiding the dilemma of having a highly-peaked policy function (𝜋(𝑎|𝑠)) towards a few actions, a known problem with on-policy models. 
* The agent we trained requires only raw frames from the screen and game states from the AI side, without using opponents’ information. Therefore, the technique applied is general and suitable for training computer agents in other environments which uses raw frames directly. 
* Our agent is capable of playing against other human players online and survive in this massively multiplayer game, and is proficient at performing tricky moves upon the encounter of enemies, including intensive sharp turns, high speed twist, and circulations.


## Environment
![](https://github.com/andi611/RL_actor_critic_slitherio_Tensorflow/blob/master/data/Environment.png)

## Model
![](https://github.com/andi611/RL_actor_critic_slitherio_Tensorflow/blob/master/data/network_architechture.png)

## Architecture
![](https://github.com/andi611/RL_actor_critic_slitherio_Tensorflow/blob/master/data/network_parameters.png)

## Screen Preprocess
![](https://github.com/andi611/RL_actor_critic_slitherio_Tensorflow/blob/master/data/screen_processing.png)

## Training Pipeline
![](https://github.com/andi611/RL_actor_critic_slitherio_Tensorflow/blob/master/data/training_pipeline.png)

## Exploration Tuning Techniques
![](https://github.com/andi611/RL_actor_critic_slitherio_Tensorflow/blob/master/data/exploration_tuning.png)

## Usage
```
python3 ./src/train_AC.py train
python3 ./src/play_AC.py train
```

## Detail
* See the following files for more detail:
```
- description.pdf
- presentation.pdf
```