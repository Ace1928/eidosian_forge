import pathlib
import unittest
from pygame import encode_string, encode_file_path
def test_string_with_null_bytes(self):
    b = b'a\x00b\x00c'
    encoded_string = encode_string(b, etype=SyntaxError)
    encoded_decode_string = encode_string(b.decode(), 'ascii', 'strict')
    self.assertIs(encoded_string, b)
    self.assertEqual(encoded_decode_string, b)