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
def update_training_data(self, board_state: np.ndarray, score: int):
    """
        Updates the training data with the given board state and score.

        Args:
            board_state (np.ndarray): The current game board state.
            score (int): The score associated with the board state.
        """
    self.training_data.append(board_state.flatten())
    self.target_scores.append(score)