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
def execute_with_input():
    with open(inputfn, 'rb') as stdin:
        p = subprocess.Popen(base_cmd, stdin=stdin, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        out, err = p.communicate()
        if p.returncode != 42:
            self.fail('unexpected return code %d\n-- stdout:\n%s\n-- stderr:\n%s\n' % (p.returncode, out, err))
        return err