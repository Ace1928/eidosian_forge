import unittest
from fastimport import (
def test_two_paths(self):
    c = helpers.common_directory([b'foo', b'bar'])
    self.assertEqual(c, b'')
    c = helpers.common_directory([b'foo/', b'bar'])
    self.assertEqual(c, b'')
    c = helpers.common_directory([b'foo/', b'foo/bar'])
    self.assertEqual(c, b'foo/')
    c = helpers.common_directory([b'foo/bar/x', b'foo/bar/y'])
    self.assertEqual(c, b'foo/bar/')
    c = helpers.common_directory([b'foo/bar/aa_x', b'foo/bar/aa_y'])
    self.assertEqual(c, b'foo/bar/')