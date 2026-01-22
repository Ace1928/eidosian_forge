import unittest
import pygame
import pygame._sdl2.controller as controller
from pygame.tests.test_utils import prompt, question
def test__auto_init(self):
    c = self._get_first_controller()
    if c:
        self.assertTrue(c.get_init())
    else:
        self.skipTest('No controller connected')