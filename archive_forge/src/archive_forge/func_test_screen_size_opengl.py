import unittest
import os
import sys
import time
import pygame, pygame.transform
from pygame.tests.test_utils import question
from pygame import display
def test_screen_size_opengl(self):
    """returns a surface with the same size requested.
        |tags:display,slow,opengl|
        """
    pygame.display.init()
    screen = pygame.display.set_mode((640, 480), pygame.OPENGL)
    self.assertEqual((640, 480), screen.get_size())