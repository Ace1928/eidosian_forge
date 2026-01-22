from typing import Callable
import gym
from gym import RewardWrapper
Transforms the reward using callable :attr:`f`.

        Args:
            reward: The reward to transform

        Returns:
            The transformed reward
        