import pygame
from Constants import *
from Menu import *
from GameController import GameController
from GA import *
import sys
def draw_snake_body(self, body):
    self.draw_rect(body, color=SNAKE_COLOR, border=True)