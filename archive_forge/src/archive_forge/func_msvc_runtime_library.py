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
def msvc_runtime_library():
    """Return name of MSVC runtime library if Python was built with MSVC >= 7"""
    ver = msvc_runtime_major()
    if ver:
        if ver < 140:
            return 'msvcr%i' % ver
        else:
            return 'vcruntime%i' % ver
    else:
        return None