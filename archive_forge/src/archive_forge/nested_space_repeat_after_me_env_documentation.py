import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete, Tuple
import numpy as np
import tree  # pip install dm_tree
from ray.rllib.utils.spaces.space_utils import flatten_space
Env for which policy has to repeat the (possibly complex) observation.

    The action space and observation spaces are always the same and may be
    arbitrarily nested Dict/Tuple Spaces.
    Rewards are given for exactly matching Discrete sub-actions and for being
    as close as possible for Box sub-actions.
    