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
def make_temp_file(suffix='', prefix='', text=True):
    if not hasattr(_tdata, 'tempdir'):
        _tdata.tempdir = tempfile.mkdtemp()
        _tmpdirs.append(_tdata.tempdir)
    fid, name = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=_tdata.tempdir, text=text)
    fo = os.fdopen(fid, 'w')
    return (fo, name)