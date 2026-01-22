import unittest
from fastimport import (
def test_lots_of_paths(self):
    c = helpers.common_directory([b'foo/bar/x', b'foo/bar/y', b'foo/bar/z'])
    self.assertEqual(c, b'foo/bar/')