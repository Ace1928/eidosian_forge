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
def omitted_child_test_wrapper(result_queue, cache_dir, second_call):
    with override_config('CACHE_DIR', cache_dir):

        @njit(cache=True)
        def test(num=1000):
            return num
        try:
            output = test()
            if second_call:
                assert test._cache_hits[test.signatures[0]] == 1, 'Cache did not hit as expected'
                assert test._cache_misses[test.signatures[0]] == 0, 'Cache has an unexpected miss'
            else:
                assert test._cache_misses[test.signatures[0]] == 1, 'Cache did not miss as expected'
                assert test._cache_hits[test.signatures[0]] == 0, 'Cache has an unexpected hit'
            success = True
        except:
            output = traceback.format_exc()
            success = False
        result_queue.put((success, output))