import unittest
import unittest.mock
import os.path
import sys
import test.support
from test.support import os_helper
from ctypes import *
from ctypes.util import find_library
def test_find_library_with_gcc(self):
    with unittest.mock.patch('ctypes.util._findSoname_ldconfig', lambda *args: None):
        self.assertNotEqual(find_library('c'), None)