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
def sorted_glob(fileglob):
    """sorts output of python glob for https://bugs.python.org/issue30461
    to allow extensions to have reproducible build results"""
    return sorted(glob.glob(fileglob))