import pathlib
import unittest
from pygame import encode_string, encode_file_path
def test_encoding_error(self):
    u = 'a\x80b'
    encoded_string = encode_string(u, 'ascii', 'strict')
    self.assertIsNone(encoded_string)