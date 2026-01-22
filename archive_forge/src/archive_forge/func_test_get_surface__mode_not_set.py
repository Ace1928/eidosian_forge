import unittest
import os
import sys
import time
import pygame, pygame.transform
from pygame.tests.test_utils import question
from pygame import display
def test_get_surface__mode_not_set(self):
    """Ensures get_surface handles the display mode not being set."""
    surface = pygame.display.get_surface()
    self.assertIsNone(surface)