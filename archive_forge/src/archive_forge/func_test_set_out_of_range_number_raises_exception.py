import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_set_out_of_range_number_raises_exception(self):
    i = 0
    data_rows = [(_int_max + 1, TypeError), (_int_max + 1e-05, TypeError), (_int_max, None), (_int_max - 1, None), (_int_max - 2, None), (_int_max - 10, None), (_int_max - 63, None), (_int_max - 64, None), (_int_max - 65, None), (_int_min - 1, TypeError), (_int_min - 1e-05, TypeError), (_int_min, None), (_int_min + 1, None), (_int_min + 2, None), (_int_min + 10, None), (_int_min + 62, None), (_int_min + 63, None), (_int_min + 64, None), (0, None), (100000, None), (-100000, None)]
    single_attribute_names = ['x', 'y', 'w', 'h', 'width', 'height', 'top', 'left', 'bottom', 'right', 'centerx', 'centery']
    tuple_value_attribute_names = ['topleft', 'topright', 'bottomleft', 'bottomright', 'midtop', 'midleft', 'midbottom', 'midright', 'size', 'center']
    for row in data_rows:
        for attribute_name in single_attribute_names:
            value, expected = row
            with self.subTest(row=row, name=f'r.{attribute_name}'):
                actual = Rect(0, 0, 0, 0)
                if expected:
                    self.assertRaises(TypeError, setattr, actual, attribute_name, value)
                else:
                    setattr(actual, attribute_name, value)
                    self.assertEqual(value, getattr(actual, attribute_name))
        other = _random_int()
        for attribute_name in tuple_value_attribute_names:
            value, expected = row
            with self.subTest(row=row, name=f'r.{attribute_name}[0]'):
                actual = Rect(0, 0, 0, 0)
                if expected:
                    self.assertRaises(TypeError, setattr, actual, attribute_name, (value, other))
                else:
                    setattr(actual, attribute_name, (value, other))
                    self.assertEqual((value, other), getattr(actual, attribute_name))
            with self.subTest(row=row, name=f'r.{attribute_name}[1]'):
                actual = Rect(0, 0, 0, 0)
                if expected:
                    self.assertRaises(TypeError, setattr, actual, attribute_name, (other, value))
                else:
                    setattr(actual, attribute_name, (other, value))
                    self.assertEqual((other, value), getattr(actual, attribute_name))