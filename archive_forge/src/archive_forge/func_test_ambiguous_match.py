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
def test_ambiguous_match(self):

    def check(args, expected, expected_reverse):
        self.assert_resolve_overload(cases, args, expected)
        self.assert_resolve_overload(cases[::-1], args, expected_reverse)
    cases = [i16(i16, i16), i32(i32, i32), f64(f64, f64)]
    check((i8, i8), cases[0], cases[1])
    check((u16, u16), cases[1], cases[2])
    cases = [i32(i32, i32), f32(f32, f32)]
    check((u32, u32), cases[0], cases[1])