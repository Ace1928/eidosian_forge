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
def clean_up_temporary_directory():
    if _tmpdirs is not None:
        for d in _tmpdirs:
            try:
                shutil.rmtree(d)
            except OSError:
                pass