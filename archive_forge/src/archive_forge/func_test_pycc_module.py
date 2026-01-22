import warnings
import unittest
from contextlib import contextmanager
from numba import jit, vectorize, guvectorize
from numba.core.errors import (NumbaDeprecationWarning,
from numba.tests.support import TestCase, needs_setuptools
@needs_setuptools
@TestCase.run_test_in_subprocess
def test_pycc_module(self):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always', category=NumbaPendingDeprecationWarning)
        import numba.pycc
        expected_str = "The 'pycc' module is pending deprecation."
        self.check_warning(w, expected_str, NumbaPendingDeprecationWarning)