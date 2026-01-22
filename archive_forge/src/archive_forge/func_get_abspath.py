from collections import namedtuple
from hashlib import sha256
import os
import shutil
import sys
import fnmatch
from sympy.testing.pytest import XFAIL
def get_abspath(path, cwd='.'):
    """ Returns the absolute path.

    Parameters
    ==========

    path : str
        (relative) path.
    cwd : str
        Path to root of relative path.
    """
    if os.path.isabs(path):
        return path
    else:
        if not os.path.isabs(cwd):
            cwd = os.path.abspath(cwd)
        return os.path.abspath(os.path.join(cwd, path))