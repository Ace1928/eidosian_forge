import numpy as np
import sys
import traceback
from numba import jit, njit
from numba.core import types, errors
from numba.tests.support import (TestCase, expected_failure_py311,
import unittest
def check_raise_nested(self, flags, **jit_args):
    """
        Check exception propagation from nested functions.
        """
    for clazz in [MyError, UDEArgsToSuper, UDENoArgSuper]:
        inner_pyfunc = raise_instance(clazz, 'some message')
        pyfunc = outer_function(inner_pyfunc)
        inner_cfunc = jit(**jit_args)(inner_pyfunc)
        cfunc = jit(**jit_args)(outer_function(inner_cfunc))
        self.check_against_python(flags, pyfunc, cfunc, clazz, 1)
        self.check_against_python(flags, pyfunc, cfunc, ValueError, 2)
        self.check_against_python(flags, pyfunc, cfunc, OtherError, 3)