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
def simulate_move(board: np.ndarray, move: str) -> Tuple[np.ndarray, int]:
    """
        Simulates a move on the board and returns the new board state and score gained.

        This function shifts the tiles in the specified direction and combines tiles of the same value.

        Args:
            board (np.ndarray): The current game board.
            move (str): The move to simulate ('up', 'down', 'left', 'right').

        Returns:
            Tuple[np.ndarray, int]: The new board state and score gained from the move.
        """

    @StandardDecorator()
    def shift_and_combine(row: list) -> Tuple[list, int]:
        """
            Shifts non-zero elements to the left and combines elements of the same value.
            Args:
                row (list): A row (or column) from the game board.
            Returns:
                Tuple[list, int]: The shifted and combined row, and the score gained.
            """
        non_zero = [i for i in row if i != 0]
        combined = []
        score = 0
        skip = False
        for i in range(len(non_zero)):
            if skip:
                skip = False
                continue
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                combined.append(2 * non_zero[i])
                score += 2 * non_zero[i]
                skip = True
            else:
                combined.append(non_zero[i])
        combined.extend([0] * (len(row) - len(combined)))
        return (combined, score)

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
    rotated_board = rotate_board(board, move)
    new_board = np.zeros_like(board)
    total_score = 0
    for i, row in enumerate(rotated_board):
        new_row, score = shift_and_combine(list(row))
        total_score += score
        new_board[i] = new_row
    if move in ['up', 'down']:
        new_board = new_board.T
    elif move == 'right':
        new_board = np.rot90(new_board, 2)
    return (new_board, total_score)