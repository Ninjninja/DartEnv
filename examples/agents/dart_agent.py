__author__ = 'yuwenhao'

import gym
import numpy as np

if __name__ == '__main__':
    env = gym.make('DartBlockPush-v1')

    env.reset()

    for i in range(1000):
        env.step([np.random.uniform(-0.5,0.5),np.random.uniform(-0.5,0.5)])
        env.render()

    env.render(close=True)