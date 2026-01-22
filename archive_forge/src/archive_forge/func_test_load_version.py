from ctypes import *
import os
import shutil
import subprocess
import sys
import unittest
import test.support
from test.support import import_helper
from test.support import os_helper
from ctypes.util import find_library
def test_load_version(self):
    if libc_name is None:
        self.skipTest('could not find libc')
    if os.path.basename(libc_name) != 'libc.so.6':
        self.skipTest('wrong libc path for test')
    cdll.LoadLibrary('libc.so.6')
    self.assertRaises(OSError, cdll.LoadLibrary, 'libc.so.9')
    self.assertRaises(OSError, cdll.LoadLibrary, self.unknowndll)