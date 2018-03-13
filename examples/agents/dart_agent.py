__author__ = 'yuwenhao'

import gym
import numpy as np

if __name__ == '__main__':
    env = gym.make('DartBlockPush-v1')

    env.reset()

    for i in range(1000):
        # print(i)
        ob, reward, done, _ = env.step([np.random.uniform(-0.5,0.5),np.random.uniform(-0.5,0.5)])
        print(done)
        if done:
            break
        env.render()

    env.render(close=True)