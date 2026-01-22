import os.path
from os.path import abspath
import re
import sys
import types
import pickle
from test import support
from test.support import import_helper
import unittest
import unittest.mock
import unittest.test
def test_get_name_from_path(self):
    loader = unittest.TestLoader()
    loader._top_level_dir = '/foo'
    name = loader._get_name_from_path('/foo/bar/baz.py')
    self.assertEqual(name, 'bar.baz')
    if not __debug__:
        return
    with self.assertRaises(AssertionError):
        loader._get_name_from_path('/bar/baz.py')