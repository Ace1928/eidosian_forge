import copy
from collections import Iterable
import numpy as np
from numba import jit, prange
from numba.typed import List
from ray.rllib.examples.env.coin_game_non_vectorized_env import CoinGame
from ray.rllib.utils import override
@jit(nopython=True)
def vectorized_step_wt_numba_optimization(actions, batch_size, red_pos, blue_pos, coin_pos, red_coin, grid_size: int, asymmetric: bool, max_steps: int, both_players_can_pick_the_same_coin: bool):
    red_pos, blue_pos = move_players(batch_size, actions, red_pos, blue_pos, grid_size)
    reward, generate, red_pick_any, red_pick_red, blue_pick_any, blue_pick_blue = compute_reward(batch_size, red_pos, blue_pos, coin_pos, red_coin, asymmetric, both_players_can_pick_the_same_coin)
    coin_pos = generate_coin(batch_size, generate, red_coin, red_pos, blue_pos, coin_pos, grid_size)
    obs = generate_observations_wt_numba_optimization(batch_size, red_pos, blue_pos, coin_pos, red_coin, grid_size)
    return (red_pos, blue_pos, reward, coin_pos, obs, red_coin, red_pick_any, red_pick_red, blue_pick_any, blue_pick_blue)