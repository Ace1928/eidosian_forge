import os
import subprocess
import sys
import warnings
import numpy as np
import unittest
from numba import jit
from numba.core.errors import (
from numba.core import errors
from numba.tests.support import ignore_internal_warnings
def test_disable_performance_warnings(self):
    not_found_ret_code = 55
    found_ret_code = 99
    expected = "'parallel=True' was specified but no transformation"
    parallel_code = 'if 1:\n            import warnings\n            from numba.tests.error_usecases import foo\n            import numba\n            from numba.tests.support import ignore_internal_warnings\n            with warnings.catch_warnings(record=True) as w:\n                warnings.simplefilter(\'always\')\n                ignore_internal_warnings()\n                foo()\n            for x in w:\n                if x.category == numba.errors.NumbaPerformanceWarning:\n                    if "%s" in str(x.message):\n                        exit(%s)\n            exit(%s)\n        ' % (expected, found_ret_code, not_found_ret_code)
    popen = subprocess.Popen([sys.executable, '-c', parallel_code])
    out, err = popen.communicate()
    self.assertEqual(popen.returncode, found_ret_code)
    env = dict(os.environ)
    env['NUMBA_DISABLE_PERFORMANCE_WARNINGS'] = '1'
    popen = subprocess.Popen([sys.executable, '-c', parallel_code], env=env)
    out, err = popen.communicate()
    self.assertEqual(popen.returncode, not_found_ret_code)