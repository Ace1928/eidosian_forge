from contextlib import closing
from io import StringIO
from os import path
from typing import List, Optional
import numpy as np
from gym import Env, logger, spaces, utils
from gym.envs.toy_text.utils import categorical_sample
from gym.error import DependencyNotInstalled
def generate_random_map(size: int=8, p: float=0.8) -> List[str]:
    """Generates a random valid map (one that has a path from start to goal)

    Args:
        size: size of each side of the grid
        p: probability that a tile is frozen

    Returns:
        A random valid map
    """
    valid = False
    board = []
    while not valid:
        p = min(1, p)
        board = np.random.choice(['F', 'H'], (size, size), p=[p, 1 - p])
        board[0][0] = 'S'
        board[-1][-1] = 'G'
        valid = is_valid(board, size)
    return [''.join(x) for x in board]