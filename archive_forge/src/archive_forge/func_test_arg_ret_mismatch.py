import os, sys, subprocess
import dis
import itertools
import numpy as np
import numba
from numba import jit, njit
from numba.core import errors, ir, types, typing, typeinfer, utils
from numba.core.typeconv import Conversion
from numba.extending import overload_method
from numba.tests.support import TestCase, tag
from numba.tests.test_typeconv import CompatibilityTestMixin
from numba.core.untyped_passes import TranslateByteCode, IRProcessing
from numba.core.typed_passes import PartialTypeInference
from numba.core.compiler_machinery import FunctionPass, register_pass
import unittest
def test_arg_ret_mismatch(self):

    def foo(x):
        return x
    args = (types.Array(i32, 1, 'C'),)
    return_type = f32
    try:
        njit(return_type(*args))(foo)
    except errors.TypingError as e:
        pass
    else:
        self.fail('Should complain about array casting to float32')