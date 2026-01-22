import os
import sys
import platform
import inspect
import traceback
import pdb
import re
import linecache
import time
from fnmatch import fnmatch
from timeit import default_timer as clock
import doctest as pdoctest  # avoid clashing with our doctest() function
from doctest import DocTestFinder, DocTestRunner
import random
import subprocess
import shutil
import signal
import stat
import tempfile
import warnings
from contextlib import contextmanager
from inspect import unwrap
from sympy.core.cache import clear_cache
from sympy.external import import_module
from sympy.external.gmpy import GROUND_TYPES, HAS_GMPY
from collections import namedtuple
def import_error(self, filename, exc_info):
    self._exceptions.append((filename, None, exc_info))
    rel_name = filename[len(self._root_dir) + 1:]
    self.write(rel_name)
    self.write('[?]   Failed to import', 'Red')
    self.write(' ')
    self.write('[FAIL]', 'Red', align='right')
    self.write('\n')