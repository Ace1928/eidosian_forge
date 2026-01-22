import os
import unittest
from unittest.mock import patch
from kivy.utils import (boundary, escape_markup, format_bytes_to_human,
from kivy import utils
def test_Platform_android(self):
    with patch.dict('os.environ', {'KIVY_BUILD': 'android'}):
        self.assertEqual(_get_platform(), 'android')
    self.assertNotIn('KIVY_BUILD', os.environ)