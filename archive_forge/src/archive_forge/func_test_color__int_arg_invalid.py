import math
import operator
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame
from pygame.colordict import THECOLORS
def test_color__int_arg_invalid(self):
    """Ensures invalid int values are detected when creating Color objects."""
    with self.assertRaises(ValueError):
        color = pygame.Color(8589934591)