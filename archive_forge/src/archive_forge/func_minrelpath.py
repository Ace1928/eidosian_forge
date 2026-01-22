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
def minrelpath(path):
    """Resolve `..` and '.' from path.
    """
    if not is_string(path):
        return path
    if '.' not in path:
        return path
    l = path.split(os.sep)
    while l:
        try:
            i = l.index('.', 1)
        except ValueError:
            break
        del l[i]
    j = 1
    while l:
        try:
            i = l.index('..', j)
        except ValueError:
            break
        if l[i - 1] == '..':
            j += 1
        else:
            del l[i], l[i - 1]
            j = 1
    if not l:
        return ''
    return os.sep.join(l)