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
def predict_move(self, board_state: np.ndarray) -> int:
    """
        Predicts the best move for a given board state using the neural network model.

        Args:
            board_state (np.ndarray): The current game board state.

        Returns:
            int: The index of the highest probability move.
        """
    prediction = self.model.predict(board_state.flatten().reshape(1, -1))
    return np.argmax(prediction)