import os
import unittest
from unittest.mock import patch
from kivy.utils import (boundary, escape_markup, format_bytes_to_human,
from kivy import utils
def test_get_hex_from_color_noalpha(self):
    actual = get_hex_from_color([0, 1, 0])
    expected = '#00ff00'
    self.assertEqual(actual, expected)