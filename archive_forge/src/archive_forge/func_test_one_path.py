import unittest
from fastimport import (
def test_one_path(self):
    c = helpers.common_directory([b'foo'])
    self.assertEqual(c, b'')
    c = helpers.common_directory([b'foo/'])
    self.assertEqual(c, b'foo/')
    c = helpers.common_directory([b'foo/bar'])
    self.assertEqual(c, b'foo/')