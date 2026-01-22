from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def create_bounding_rect(points):
    """Creates a bounding rect from the given points."""
    xmin = xmax = points[0][0]
    ymin = ymax = points[0][1]
    for x, y in points[1:]:
        xmin = min(x, xmin)
        xmax = max(x, xmax)
        ymin = min(y, ymin)
        ymax = max(y, ymax)
    return pygame.Rect((xmin, ymin), (xmax - xmin + 1, ymax - ymin + 1))