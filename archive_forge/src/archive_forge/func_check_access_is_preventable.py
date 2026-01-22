import inspect
import llvmlite.binding as ll
import multiprocessing
import numpy as np
import os
import stat
import shutil
import subprocess
import sys
import traceback
import unittest
import warnings
from numba import njit
from numba.core import codegen
from numba.core.caching import _UserWideCacheLocator
from numba.core.errors import NumbaWarning
from numba.parfors import parfor
from numba.tests.support import (
from numba import njit
from numba import njit
from file2 import function2
from numba import njit
def check_access_is_preventable():
    tempdir = temp_directory('test_cache')
    test_dir = os.path.join(tempdir, 'writable_test')
    os.mkdir(test_dir)
    with open(os.path.join(test_dir, 'write_ok'), 'wt') as f:
        f.write('check1')
    os.chmod(test_dir, 320)
    try:
        with open(os.path.join(test_dir, 'write_forbidden'), 'wt') as f:
            f.write('check2')
        return False
    except PermissionError:
        return True
    finally:
        os.chmod(test_dir, 509)
        shutil.rmtree(test_dir)