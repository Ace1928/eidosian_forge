import unittest
import os
import sys
import time
import pygame, pygame.transform
from pygame.tests.test_utils import question
from pygame import display
def test_x11_set_mode_crash_gh1654(self):
    pygame.display.init()
    pygame.display.quit()
    screen = pygame.display.set_mode((640, 480), 0)
    self.assertEqual((640, 480), screen.get_size())