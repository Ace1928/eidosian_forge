import math
import operator
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame
from pygame.colordict import THECOLORS
def test_color__sequence_arg_invalid_format(self):
    """Ensures invalid sequences are detected when creating Color objects
        with the wrong number of values.
        """
    cls = pygame.Color
    for seq_type in (tuple, list):
        self.assertRaises(ValueError, cls, seq_type((100,)))
        self.assertRaises(ValueError, cls, seq_type((100, 90)))
        self.assertRaises(ValueError, cls, seq_type((100, 90, 80, 70, 60)))