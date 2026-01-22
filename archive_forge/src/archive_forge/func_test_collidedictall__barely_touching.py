import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_collidedictall__barely_touching(self):
    """Ensures collidedictall works correctly for rects that barely touch."""
    rect = Rect(1, 1, 10, 10)
    collide_rect = Rect(0, 0, 1, 1)
    collide_item1 = ('collide 1', collide_rect)
    no_collide_item1 = ('no collide 1', Rect(50, 50, 20, 20))
    no_collide_item2 = ('no collide 2', Rect(60, 60, 20, 20))
    no_collide_item3 = ('no collide 3', Rect(70, 70, 20, 20))
    no_collide_rect_values = dict((no_collide_item1, no_collide_item2, no_collide_item3))
    no_collide_rect_keys = {tuple(v): k for k, v in no_collide_rect_values.items()}
    for attr in ('topleft', 'topright', 'bottomright', 'bottomleft'):
        setattr(collide_rect, attr, getattr(rect, attr))
        for use_values in (True, False):
            if use_values:
                expected_items = [collide_item1]
                d = dict(no_collide_rect_values)
            else:
                expected_items = [(tuple(collide_item1[1]), collide_item1[0])]
                d = dict(no_collide_rect_keys)
            d.update(expected_items)
            collide_items = rect.collidedictall(d, use_values)
            self._assertCountEqual(collide_items, expected_items)