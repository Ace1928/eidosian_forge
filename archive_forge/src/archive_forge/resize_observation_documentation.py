from typing import Union
import numpy as np
import gym
from gym.error import DependencyNotInstalled
from gym.spaces import Box
Updates the observations by resizing the observation to shape given by :attr:`shape`.

        Args:
            observation: The observation to reshape

        Returns:
            The reshaped observations

        Raises:
            DependencyNotInstalled: opencv-python is not installed
        