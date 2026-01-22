import sys
import subprocess
import numpy as np
import os
import warnings
from numba import jit, njit, types
from numba.core import errors
from numba.experimental import structref
from numba.extending import (overload, intrinsic, overload_method,
from numba.core.compiler import CompilerBase
from numba.core.untyped_passes import (TranslateByteCode, FixupArgs,
from numba.core.typed_passes import (NopythonTypeInference, DeadCodeElimination,
from numba.core.compiler_machinery import PassManager
from numba.core.types.functions import _err_reasons as error_reasons
from numba.tests.support import (skip_parfors_unsupported, override_config,
import unittest
@skip_unless_scipy
def test_error_function_source_is_correct(self):
    """ Checks that the reported source location for an overload is the
        overload implementation source, not the actual function source from the
        target library."""

    @njit
    def foo():
        np.linalg.svd('chars')
    with self.assertRaises(errors.TypingError) as raises:
        foo()
    excstr = str(raises.exception)
    self.assertIn(error_reasons['specific_error'].splitlines()[0], excstr)
    expected_file = os.path.join('numba', 'np', 'linalg.py')
    expected = f"Overload in function 'svd_impl': File: {expected_file}:"
    self.assertIn(expected.format(expected_file), excstr)