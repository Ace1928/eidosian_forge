from gymnasium.spaces import Box
import numpy as np
from gymnasium.envs.classic_control import CartPoleEnv
Partially observable variant of the CartPole gym environment.

    https://github.com/openai/gym/blob/master/gym/envs/classic_control/
    cartpole.py

    We delete the x- and angular velocity components of the state, so that it
    can only be solved by a memory enhanced model (policy).
    