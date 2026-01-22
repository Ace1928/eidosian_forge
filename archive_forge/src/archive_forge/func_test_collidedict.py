import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_collidedict(self):
    """Ensures collidedict detects collisions."""
    rect = Rect(1, 1, 10, 10)
    collide_item1 = ('collide 1', rect.copy())
    collide_item2 = ('collide 2', Rect(5, 5, 10, 10))
    no_collide_item1 = ('no collide 1', Rect(60, 60, 10, 10))
    no_collide_item2 = ('no collide 2', Rect(70, 70, 10, 10))
    rect_values = dict((collide_item1, collide_item2, no_collide_item1, no_collide_item2))
    value_collide_items = (collide_item1, collide_item2)
    rect_keys = {tuple(v): k for k, v in rect_values.items()}
    key_collide_items = tuple(((tuple(v), k) for k, v in value_collide_items))
    for use_values in (True, False):
        if use_values:
            expected_items = value_collide_items
            d = rect_values
        else:
            expected_items = key_collide_items
            d = rect_keys
        collide_item = rect.collidedict(d, use_values)
        self.assertIn(collide_item, expected_items)