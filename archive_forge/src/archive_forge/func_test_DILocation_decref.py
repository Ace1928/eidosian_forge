from collections import namedtuple
import inspect
import re
import numpy as np
import math
from textwrap import dedent
import unittest
import warnings
from numba.tests.support import (TestCase, override_config,
from numba import jit, njit
from numba.core import types
from numba.core.datamodel import default_manager
from numba.core.errors import NumbaDebugInfoWarning
import llvmlite.binding as llvm
@TestCase.run_test_in_subprocess(envvars=_NUMBA_OPT_0_ENV)
def test_DILocation_decref(self):
    """ This tests that decref's generated from `ir.Del`s as variables go
        out of scope do not have debuginfo associated with them (the location of
        `ir.Del` is an implementation detail).
        """

    @njit(debug=True)
    def sink(*x):
        pass

    @njit(debug=True)
    def foo(a):
        x = (a, a)
        if a[0] == 0:
            sink(x)
            return 12
        z = x[0][0]
        return z
    sig = (types.float64[::1],)
    full_ir = self._get_llvmir(foo, sig=sig)
    count = 0
    for line in full_ir.splitlines():
        line_stripped = line.strip()
        if line_stripped.startswith('call void @NRT_decref'):
            self.assertRegex(line, '.*meminfo\\.[0-9]+\\)$')
            count += 1
    self.assertGreater(count, 0)