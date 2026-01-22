import pygame
from Constants import *
from Menu import *
from GameController import GameController
from GA import *
import sys
def is_quit(self, event):
    if event.type == pygame.QUIT:
        self.running, self.playing = (False, False)
        self.curr_menu.run_display = False
        return True
    return False