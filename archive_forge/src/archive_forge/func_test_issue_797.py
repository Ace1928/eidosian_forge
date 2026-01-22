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
def test_issue_797(self):
    """https://github.com/numba/numba/issues/797#issuecomment-58592401

        Undeterministic triggering of tuple coercion error
        """
    foo = jit(nopython=True)(issue_797)
    g = np.zeros(shape=(10, 10), dtype=np.int32)
    foo(np.int32(0), np.int32(0), np.int32(1), np.int32(1), g)