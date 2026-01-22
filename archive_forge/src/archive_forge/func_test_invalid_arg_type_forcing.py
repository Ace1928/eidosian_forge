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
def test_invalid_arg_type_forcing(self):

    def foo(iters):
        a = range(iters)
        return iters
    args = (u32,)
    return_type = u8
    cfunc = njit(return_type(*args))(foo)
    cres = cfunc.overloads[args]
    typemap = cres.type_annotation.typemap
    self.assertEqual(typemap['iters'], u32)