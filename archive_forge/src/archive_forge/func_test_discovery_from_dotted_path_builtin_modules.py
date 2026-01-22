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
def test_discovery_from_dotted_path_builtin_modules(self):
    loader = unittest.TestLoader()
    listdir = os.listdir
    os.listdir = lambda _: ['test_this_does_not_exist.py']
    isfile = os.path.isfile
    isdir = os.path.isdir
    os.path.isdir = lambda _: False
    orig_sys_path = sys.path[:]

    def restore():
        os.path.isfile = isfile
        os.path.isdir = isdir
        os.listdir = listdir
        sys.path[:] = orig_sys_path
    self.addCleanup(restore)
    with self.assertRaises(TypeError) as cm:
        loader.discover('sys')
    self.assertEqual(str(cm.exception), 'Can not use builtin modules as dotted module names')