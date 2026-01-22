from ai_logic import (
from gui_utils import (
from game_manager import (
from typing import List, Tuple
import types
import importlib.util
import logging
import numpy as np
@StandardDecorator()
def main_game_loop():
    board = initialize_game()
    score = 0
    game_over = False
    while not game_over:
        best_move = calculate_best_move(board)
        if best_move:
            board, move_score = simulate_move(board, best_move)
            score += move_score
            add_random_tile(board)
            update_gui(board, score)
            game_over = is_game_over(board)
        else:
            game_over = True
    print(f'Game Over! Final Score: {score}')