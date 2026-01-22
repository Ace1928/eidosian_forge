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
def realpath(path):
    if path == os.path.join(mod_dir, 'foo.py'):
        return os.path.join(expected_dir, 'foo.py')
    return path