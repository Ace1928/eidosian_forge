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
def assert_unify(self, aty, bty, expected):
    ctx = typing.Context()
    template = '{0}, {1} -> {2} != {3}'
    for unify_func in (ctx.unify_types, ctx.unify_pairs):
        unified = unify_func(aty, bty)
        self.assertEqual(unified, expected, msg=template.format(aty, bty, unified, expected))
        unified = unify_func(bty, aty)
        self.assertEqual(unified, expected, msg=template.format(bty, aty, unified, expected))