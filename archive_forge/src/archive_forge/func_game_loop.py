import pygame
from Constants import *
from Menu import *
from GameController import GameController
from GA import *
import sys
def game_loop(self):
    while self.playing:
        self.event_handler()
        if self.BACK:
            self.playing = False
        self.display.fill(WINDOW_COLOR)
        if self.controller.algo != None:
            self.draw_elements()
        self.window.blit(self.display, (0, 0))
        pygame.display.update()
        self.clock.tick(60)
        self.reset_keys()