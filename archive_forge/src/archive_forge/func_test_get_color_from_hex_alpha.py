import os
import unittest
from unittest.mock import patch
from kivy.utils import (boundary, escape_markup, format_bytes_to_human,
from kivy import utils
def test_get_color_from_hex_alpha(self):
    actual = get_color_from_hex('#00FF7F7F')
    expected = [0.0, 1.0, 0.49803921, 0.49803921]
    for i in range(4):
        self.assertAlmostEqual(actual[i], expected[i])