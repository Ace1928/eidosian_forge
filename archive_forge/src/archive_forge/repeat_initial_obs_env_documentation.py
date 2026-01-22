import gymnasium as gym
from gymnasium.spaces import Discrete
import random
Env in which the initial observation has to be repeated all the time.

    Runs for n steps.
    r=1 if action correct, -1 otherwise (max. R=100).
    