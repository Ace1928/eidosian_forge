import os
import unittest
from unittest.mock import patch
from kivy.utils import (boundary, escape_markup, format_bytes_to_human,
from kivy import utils
def test_interpolate_solo(self):
    values = [10.0, 19.0, 27.1]
    a = 0.0
    for i in range(0, 3):
        a = interpolate(a, 100)
        self.assertEqual(a, values[i])