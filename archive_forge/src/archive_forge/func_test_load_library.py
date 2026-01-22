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
@unittest.skipUnless(os.name == 'nt', 'test specific to Windows')
def test_load_library(self):
    if test.support.verbose:
        print(find_library('kernel32'))
        print(find_library('user32'))
    if os.name == 'nt':
        windll.kernel32.GetModuleHandleW
        windll['kernel32'].GetModuleHandleW
        windll.LoadLibrary('kernel32').GetModuleHandleW
        WinDLL('kernel32').GetModuleHandleW
        self.assertRaises(ValueError, windll.LoadLibrary, 'kernel32\x00')