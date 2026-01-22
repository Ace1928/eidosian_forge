import numpy as np
from ai_logic import (
from gui_utils import (
from typing import List, Tuple
import types
import importlib.util
import logging
import random
@StandardDecorator()
def on_game_end(board: np.ndarray, score: int):
    """
    Performs cleanup tasks when the game ends.

    Args:
        board (np.ndarray): The final game board state.
        score (int): The final score of the game.
    """