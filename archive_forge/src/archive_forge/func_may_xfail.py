from collections import namedtuple
from hashlib import sha256
import os
import shutil
import sys
import fnmatch
from sympy.testing.pytest import XFAIL
def may_xfail(func):
    if sys.platform.lower() == 'darwin' or os.name == 'nt':
        return XFAIL(func)
    else:
        return func