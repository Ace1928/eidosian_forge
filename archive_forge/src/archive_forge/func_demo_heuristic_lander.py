import math
import warnings
from typing import TYPE_CHECKING, Optional
import numpy as np
import gym
from gym import error, spaces
from gym.error import DependencyNotInstalled
from gym.utils import EzPickle, colorize
from gym.utils.step_api_compatibility import step_api_compatibility
def demo_heuristic_lander(env, seed=None, render=False):
    total_reward = 0
    steps = 0
    s, info = env.reset(seed=seed)
    while True:
        a = heuristic(env, s)
        s, r, terminated, truncated, info = step_api_compatibility(env.step(a), True)
        total_reward += r
        if render:
            still_open = env.render()
            if still_open is False:
                break
        if steps % 20 == 0 or terminated or truncated:
            print('observations:', ' '.join([f'{x:+0.2f}' for x in s]))
            print(f'step {steps} total_reward {total_reward:+0.2f}')
        steps += 1
        if terminated or truncated:
            break
    if render:
        env.close()
    return total_reward