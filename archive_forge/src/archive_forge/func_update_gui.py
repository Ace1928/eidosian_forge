from typing import List, Tuple
import types
import importlib.util
import logging
import numpy as np
from math import log2
import pygame
@StandardDecorator()
def update_gui(board: np.ndarray, score: int) -> None:
    """
    Updates the GUI with the current game state.

    Args:
        board (np.ndarray): The game board as a 2D NumPy array.
        score (int): The current score.
    """
    draw_board(board)
    draw_gui(board)
    draw_score(score)