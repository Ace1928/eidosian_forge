import math
import operator
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame
from pygame.colordict import THECOLORS
def test_color__sequence_arg_invalid_value(self):
    """Ensures invalid sequences are detected when creating Color objects."""
    cls = pygame.Color
    for seq_type in (tuple, list):
        self.assertRaises(ValueError, cls, seq_type((256, 90, 80, 70)))
        self.assertRaises(ValueError, cls, seq_type((100, 256, 80, 70)))
        self.assertRaises(ValueError, cls, seq_type((100, 90, 256, 70)))
        self.assertRaises(ValueError, cls, seq_type((100, 90, 80, 256)))