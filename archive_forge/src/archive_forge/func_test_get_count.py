import unittest
import pygame
import pygame._sdl2.controller as controller
from pygame.tests.test_utils import prompt, question
def test_get_count(self):
    self.assertGreaterEqual(controller.get_count(), 0)