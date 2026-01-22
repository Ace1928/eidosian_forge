import os
import tempfile
import textwrap
import unittest
from bpython import config
def test_load_theme(self):
    color_scheme = dict()
    config.load_theme(TEST_THEME_PATH, color_scheme, dict())
    expected = {'keyword': 'y'}
    self.assertEqual(color_scheme, expected)
    defaults = {'name': 'c'}
    expected.update(defaults)
    config.load_theme(TEST_THEME_PATH, color_scheme, defaults)
    self.assertEqual(color_scheme, expected)