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
def test_irregularly_indented_source(self):

    @njit(debug=True)
    def foo():
        return 1
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always', NumbaDebugInfoWarning)
        ignore_internal_warnings()
        foo()
    self.assertEqual(len(w), 0)
    metadata = self._get_metadata(foo, foo.signatures[0])
    lines = self._get_lines_from_debuginfo(metadata)
    self.assertEqual(len(lines), 1)