import sys
import os
import pytest
from os.path import join as pathjoin, isfile, dirname
import subprocess
import numpy as np
from numpy.testing import assert_equal, IS_WASM
def find_f2py_commands():
    if sys.platform == 'win32':
        exe_dir = dirname(sys.executable)
        if exe_dir.endswith('Scripts'):
            return [os.path.join(exe_dir, 'f2py')]
        else:
            return [os.path.join(exe_dir, 'Scripts', 'f2py')]
    else:
        version = sys.version_info
        major = str(version.major)
        minor = str(version.minor)
        return ['f2py', 'f2py' + major, 'f2py' + major + '.' + minor]