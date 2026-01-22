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
def is_local_src_dir(directory):
    """Return true if directory is local directory.
    """
    if not is_string(directory):
        return False
    abs_dir = os.path.abspath(directory)
    c = os.path.commonprefix([os.getcwd(), abs_dir])
    new_dir = abs_dir[len(c):].split(os.sep)
    if new_dir and (not new_dir[0]):
        new_dir = new_dir[1:]
    if new_dir and new_dir[0] == 'build':
        return False
    new_dir = os.sep.join(new_dir)
    return os.path.isdir(new_dir)