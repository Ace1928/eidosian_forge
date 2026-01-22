import unittest
import pygame
import pygame._sdl2.controller as controller
from pygame.tests.test_utils import prompt, question
def test_as_joystick(self):
    c = self._get_first_controller()
    if c:
        joy = c.as_joystick()
        self.assertIsInstance(joy, type(pygame.joystick.Joystick(0)))
    else:
        self.skipTest('No controller connected')