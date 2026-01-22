import os
import sys
import errno
import shutil
import random
import glob
import warnings
from IPython.utils.process import system
def shellglob(args):
    """
    Do glob expansion for each element in `args` and return a flattened list.

    Unmatched glob pattern will remain as-is in the returned list.

    """
    expanded = []
    unescape = unescape_glob if sys.platform != 'win32' else lambda x: x
    for a in args:
        expanded.extend(glob.glob(a) or [unescape(a)])
    return expanded