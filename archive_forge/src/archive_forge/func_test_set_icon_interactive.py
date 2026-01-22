import unittest
import os
import sys
import time
import pygame, pygame.transform
from pygame.tests.test_utils import question
from pygame import display
def test_set_icon_interactive(self):
    os.environ['SDL_VIDEO_WINDOW_POS'] = '100,250'
    pygame.display.quit()
    pygame.display.init()
    test_icon = pygame.Surface((32, 32))
    test_icon.fill((255, 0, 0))
    pygame.display.set_icon(test_icon)
    screen = pygame.display.set_mode((400, 100))
    pygame.display.set_caption('Is the window icon a red square?')
    response = question('Is the display icon red square?')
    self.assertTrue(response)
    pygame.display.quit()