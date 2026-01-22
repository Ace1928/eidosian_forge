from gymnasium.spaces import Box
import numpy as np
from gymnasium.envs.classic_control import PendulumEnv
Partially observable variant of the Pendulum gym environment.

    https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/
    classic_control/pendulum.py

    We delete the angular velocity component of the state, so that it
    can only be solved by a memory enhanced model (policy).
    