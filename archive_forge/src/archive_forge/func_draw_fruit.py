import pygame
from Constants import *
from Menu import *
from GameController import GameController
from GA import *
import sys
def draw_fruit(self, fruit):
    x = int(fruit.x * CELL_SIZE)
    y = int(fruit.y * CELL_SIZE)
    fruit_rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(self.display, FRUIT_COLOR, fruit_rect)