import os
import re
import sys
import copy
import glob
import atexit
import tempfile
import subprocess
import shutil
import multiprocessing
import textwrap
import importlib.util
from threading import local as tlocal
from functools import reduce
import distutils
from distutils.errors import DistutilsError
def njoin(*path):
    """Join two or more pathname components +
    - convert a /-separated pathname to one using the OS's path separator.
    - resolve `..` and `.` from path.

    Either passing n arguments as in njoin('a','b'), or a sequence
    of n names as in njoin(['a','b']) is handled, or a mixture of such arguments.
    """
    paths = []
    for p in path:
        if is_sequence(p):
            paths.append(njoin(*p))
        else:
            assert is_string(p)
            paths.append(p)
    path = paths
    if not path:
        joined = ''
    else:
        joined = os.path.join(*path)
    if os.path.sep != '/':
        joined = joined.replace('/', os.path.sep)
    return minrelpath(joined)