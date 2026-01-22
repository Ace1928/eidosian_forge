import math
from typing import Optional, Union
import numpy as np
import gym
from gym import spaces
from gym.envs.box2d.car_dynamics import Car
from gym.error import DependencyNotInstalled, InvalidAction
from gym.utils import EzPickle
def render_if_min(value, points, color):
    if abs(value) > 0.0001:
        pygame.draw.polygon(self.surf, points=points, color=color)