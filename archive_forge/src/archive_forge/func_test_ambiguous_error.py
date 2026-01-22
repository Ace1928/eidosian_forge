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
def test_ambiguous_error(self):
    ctx = typing.Context()
    cases = [i16(i16, i16), i32(i32, i32)]
    with self.assertRaises(TypeError) as raises:
        ctx.resolve_overload('foo', cases, (i8, i8), {}, allow_ambiguous=False)
    self.assertEqual(str(raises.exception).splitlines(), ['Ambiguous overloading for foo (int8, int8):', '(int16, int16) -> int16', '(int32, int32) -> int32'])