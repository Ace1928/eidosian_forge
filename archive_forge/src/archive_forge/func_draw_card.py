import os
from typing import Optional
import numpy as np
import gym
from gym import spaces
from gym.error import DependencyNotInstalled
def draw_card(np_random):
    return int(np_random.choice(deck))