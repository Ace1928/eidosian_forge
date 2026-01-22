import numpy as np  # Assuming NumPy is used for efficient array manipulation
import random
import types
import importlib.util
import logging
import collections
from typing import Deque, Dict, Tuple, List
from typing import List, Tuple
from functools import wraps
import logging
@StandardDecorator()
def rotate_board(board: np.ndarray, move: str) -> np.ndarray:
    """
            Rotates the board to simplify shifting logic.
            Args:
                board (np.ndarray): The game board.
                move (str): The move direction.
            Returns:
                np.ndarray: The rotated board.
            """
    if move == 'up':
        return board.T
    elif move == 'down':
        return np.rot90(board, 2).T
    elif move == 'left':
        return board
    elif move == 'right':
        return np.rot90(board, 2)
    else:
        raise ValueError('Invalid move direction')