import pygame
from Constants import *
from Menu import *
from GameController import GameController
from GA import *
import sys
def draw_game_stats(self):
    if self.curr_menu.state != 'GA':
        instruction = 'Space to view Ai path, W to speed up, Q to go back'
    elif self.controller.model_loaded:
        instruction = 'W to speed up, Q to go back'
    else:
        instruction = 'Space to hide all snakes, W to speed up, Q to go back'
        curr_gen = str(self.controller.curr_gen())
        best_score = str(self.controller.best_GA_score())
        stats_gen = f'Generation: {curr_gen}/{GA.generation}'
        stats_score = f'Best score: {best_score}'
        stats_hidden_node = f'Hidden nodes {Population.hidden_node}'
        self.draw_text(stats_gen, size=20, x=3 * CELL_SIZE, y=CELL_SIZE - 10)
        self.draw_text(stats_score, size=20, x=3 * CELL_SIZE, y=CELL_SIZE + 20)
        self.draw_text(stats_hidden_node, size=20, x=self.SIZE / 2, y=CELL_SIZE - 30, color=SNAKE_COLOR)
    self.draw_text(instruction, size=20, x=self.SIZE / 2, y=CELL_SIZE * NO_OF_CELLS - NO_OF_CELLS, color=WHITE)
    self.draw_text(self.curr_menu.state, size=30, x=self.SIZE / 2, y=CELL_SIZE)