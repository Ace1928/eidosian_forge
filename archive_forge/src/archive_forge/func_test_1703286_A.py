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
@unittest.skipUnless(os.name == 'nt', 'Windows-specific test')
def test_1703286_A(self):
    from _ctypes import LoadLibrary, FreeLibrary
    handle = LoadLibrary('advapi32')
    FreeLibrary(handle)