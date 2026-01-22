import gymnasium as gym
import numpy as np
import random
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.view_requirement import ViewRequirement
def successor(x):
    if isinstance(self.observation_space, gym.spaces.Discrete):
        if x == ROCK:
            return PAPER
        elif x == PAPER:
            return SCISSORS
        elif x == SCISSORS:
            return ROCK
        else:
            return random.choice([ROCK, PAPER, SCISSORS])
    elif x[ROCK] == 1:
        return PAPER
    elif x[PAPER] == 1:
        return SCISSORS
    elif x[SCISSORS] == 1:
        return ROCK
    elif x[-1] == 1:
        return random.choice([ROCK, PAPER, SCISSORS])