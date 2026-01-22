import pathlib
import unittest
from pygame import encode_string, encode_file_path
def test_etype__invalid(self):
    """Ensures invalid etypes are properly handled."""
    for etype in ('SyntaxError', self):
        self.assertRaises(TypeError, encode_file_path, 'test', etype)