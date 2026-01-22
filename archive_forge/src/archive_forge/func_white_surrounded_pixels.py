import math
import unittest
import sys
import warnings
import pygame
from pygame import draw
from pygame import draw_py
from pygame.locals import SRCALPHA
from pygame.tests import test_utils
from pygame.math import Vector2
def white_surrounded_pixels(x, y):
    offsets = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    WHITE = (255, 255, 255, 255)
    return len([1 for dx, dy in offsets if surf.get_at((x + dx, y + dy)) == WHITE])