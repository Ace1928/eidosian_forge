import numpy as np
from ai_logic import (
from gui_utils import (
from typing import List, Tuple
import types
import importlib.util
import logging
import random
@StandardDecorator()
def on_game_update(board: np.ndarray, score: int):
    """
    Performs tasks when the game state is updated.

    Args:
        board (np.ndarray): The current game board state.
        score (int): The current score of the game.
    """