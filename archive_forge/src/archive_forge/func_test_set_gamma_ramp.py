import unittest
import os
import sys
import time
import pygame, pygame.transform
from pygame.tests.test_utils import question
from pygame import display
def test_set_gamma_ramp(self):
    os.environ['SDL_VIDEO_WINDOW_POS'] = '100,250'
    pygame.display.quit()
    pygame.display.init()
    screen = pygame.display.set_mode((400, 100))
    screen.fill((100, 100, 100))
    blue_ramp = [x * 256 for x in range(0, 256)]
    blue_ramp[100] = 150 * 256
    normal_ramp = [x * 256 for x in range(0, 256)]
    gamma_success = False
    if pygame.display.set_gamma_ramp(normal_ramp, normal_ramp, blue_ramp):
        pygame.display.update()
        gamma_success = True
    if gamma_success:
        response = question('Is the window background tinted blue?')
        self.assertTrue(response)
        pygame.display.set_gamma_ramp(normal_ramp, normal_ramp, normal_ramp)
    pygame.display.quit()