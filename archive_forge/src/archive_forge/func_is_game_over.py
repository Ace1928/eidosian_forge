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
def is_game_over(board: np.ndarray) -> bool:
    """
    Checks if the game is over (no moves left).

    Args:
        board (np.ndarray): The game board.

    Returns:
        bool: True if the game is over, False otherwise.
    """
    logging.debug('Checking if the game is over.')
    game_over = not any((DeepLearningDecisionMaker.simulate_move(board, move)[0].tolist() != board.tolist() for move in ['up', 'down', 'left', 'right']))
    logging.debug(f'Game over status: {game_over}.')
    return game_over