import pathlib
import unittest
from pygame import encode_string, encode_file_path
def test_etype(self):
    b = b'a\x00b\x00c'
    self.assertRaises(TypeError, encode_file_path, b, TypeError)