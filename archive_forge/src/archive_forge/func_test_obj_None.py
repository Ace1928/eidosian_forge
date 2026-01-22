import pathlib
import unittest
from pygame import encode_string, encode_file_path
def test_obj_None(self):
    encoded_string = encode_string(None)
    self.assertIsNone(encoded_string)