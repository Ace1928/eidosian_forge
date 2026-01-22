import pygame
from Constants import *
from GA import *
import sys
def load_GA(self):
    self.game.curr_menu = self.game.main_menu
    self.run_display = False
    self.game.curr_menu.state = 'GA'
    self.game.playing = True
    self.game.load_model = True