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
def general_source_files(top_path):
    pruned_directories = {'CVS': 1, '.svn': 1, 'build': 1}
    prune_file_pat = re.compile('(?:[~#]|\\.py[co]|\\.o)$')
    for dirpath, dirnames, filenames in os.walk(top_path, topdown=True):
        pruned = [d for d in dirnames if d not in pruned_directories]
        dirnames[:] = pruned
        for f in filenames:
            if not prune_file_pat.search(f):
                yield os.path.join(dirpath, f)