import os
from typing import Optional
import numpy as np
import gym
from gym import spaces
from gym.error import DependencyNotInstalled
def usable_ace(hand):
    return 1 in hand and sum(hand) + 10 <= 21