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
def test_bound_function_error_string(self):

    def foo(x):
        x.max(-1)
    with override_config('DEVELOPER_MODE', 1):
        with self.assertRaises(errors.TypingError) as raises:
            njit('void(int64[:,:])')(foo)
    excstr = str(raises.exception)
    self.assertIn('too many positional arguments', excstr)