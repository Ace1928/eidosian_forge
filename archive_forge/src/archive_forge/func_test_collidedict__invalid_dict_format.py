import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_collidedict__invalid_dict_format(self):
    """Ensures collidedict correctly handles invalid dict parameters."""
    rect = Rect(0, 0, 10, 10)
    invalid_value_dict = ('collide', rect.copy())
    invalid_key_dict = (tuple(invalid_value_dict[1]), invalid_value_dict[0])
    for use_values in (True, False):
        d = invalid_value_dict if use_values else invalid_key_dict
        with self.assertRaises(TypeError):
            collide_item = rect.collidedict(d, use_values)