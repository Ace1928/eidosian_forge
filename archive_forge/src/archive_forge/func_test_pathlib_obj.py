import pathlib
import unittest
from pygame import encode_string, encode_file_path
def test_pathlib_obj(self):
    """Test loading string representation of pathlib object"""
    '\n        We do this because pygame functions internally use pg_EncodeString\n        to decode the filenames passed to them. So if we test that here, we\n        can safely assume that all those functions do not have any issues\n        with pathlib objects\n        '
    encoded = encode_string(pathlib.PurePath('foo'), 'utf-8')
    self.assertEqual(encoded, b'foo')
    encoded = encode_string(pathlib.Path('baz'))
    self.assertEqual(encoded, b'baz')