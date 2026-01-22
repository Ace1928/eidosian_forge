import os
import unittest
from unittest.mock import patch
from kivy.utils import (boundary, escape_markup, format_bytes_to_human,
from kivy import utils
def test_get_random_color_fixed_alpha(self):
    actual = get_random_color()
    self.assertEqual(len(actual), 4)
    self.assertEqual(actual[3], 1.0)
    actual = get_random_color(alpha=0.5)
    self.assertEqual(len(actual), 4)
    self.assertEqual(actual[3], 0.5)