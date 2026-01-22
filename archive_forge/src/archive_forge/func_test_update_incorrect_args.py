import unittest
import os
import sys
import time
import pygame, pygame.transform
from pygame.tests.test_utils import question
from pygame import display
def test_update_incorrect_args(self):
    """raises a ValueError when inputs are wrong."""
    with self.assertRaises(ValueError):
        pygame.display.update(100, 'asdf', 100, 100)
    with self.assertRaises(ValueError):
        pygame.display.update([100, 'asdf', 100, 100])