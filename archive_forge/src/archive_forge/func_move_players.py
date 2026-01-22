import copy
from collections import Iterable
import numpy as np
from numba import jit, prange
from numba.typed import List
from ray.rllib.examples.env.coin_game_non_vectorized_env import CoinGame
from ray.rllib.utils import override
@jit(nopython=True)
def move_players(batch_size, actions, red_pos, blue_pos, grid_size):
    moves = List([np.array([0, 1]), np.array([0, -1]), np.array([1, 0]), np.array([-1, 0])])
    for j in prange(batch_size):
        red_pos[j] = (red_pos[j] + moves[actions[j, 0]]) % grid_size
        blue_pos[j] = (blue_pos[j] + moves[actions[j, 1]]) % grid_size
    return (red_pos, blue_pos)