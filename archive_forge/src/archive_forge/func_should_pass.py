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
def should_pass(command):
    with self.subTest(command):
        subprocess.check_output([sys.executable, '-c', 'from ctypes import *; import nt;' + command], cwd=tmp)