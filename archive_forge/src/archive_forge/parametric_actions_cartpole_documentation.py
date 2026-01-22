import random
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict, Discrete
Same as the above ParametricActionsCartPole.

    However, action embeddings are not published inside observations,
    but will be learnt by the model.

    At each step, we emit a dict of:
        - the actual cart observation
        - a mask of valid actions (e.g., [0, 0, 1, 0, 0, 1] for 6 max avail)
        - action embeddings (w/ "dummy embedding" for invalid actions) are
          outsourced in the model and will be learned.
    