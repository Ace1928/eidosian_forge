import unittest
import platform
from numpy import (
import pygame
from pygame.locals import *
import pygame.surfarray
def do_pixels3d(surf):
    pygame.surfarray.pixels3d(surf)