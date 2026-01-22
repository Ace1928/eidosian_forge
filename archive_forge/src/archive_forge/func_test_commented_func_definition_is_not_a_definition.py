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
def test_commented_func_definition_is_not_a_definition(self):

    def foo_commented():
        raise Exception('test_string')

    def foo_docstring():
        """ def docstring containing def might match function definition!"""
        raise Exception('test_string')
    for func in (foo_commented, foo_docstring):
        with self.assertRaises(Exception) as raises:
            func()
        self.assertIn('test_string', str(raises.exception))