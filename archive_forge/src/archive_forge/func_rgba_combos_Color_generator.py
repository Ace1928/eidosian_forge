import math
import operator
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame
from pygame.colordict import THECOLORS
def rgba_combos_Color_generator():
    for rgba in rgba_combinations:
        yield pygame.Color(*rgba)