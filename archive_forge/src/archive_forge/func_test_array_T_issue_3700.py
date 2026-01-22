import itertools
import numpy as np
import sys
from collections import namedtuple
from io import StringIO
from numba import njit, typeof, prange
from numba.core import (
from numba.tests.support import (TestCase, tag, skip_parfors_unsupported,
from numba.parfors.array_analysis import EquivSet, ArrayAnalysis
from numba.core.compiler import Compiler, Flags, PassManager
from numba.core.ir_utils import remove_dead
from numba.core.untyped_passes import (ExtractByteCode, TranslateByteCode, FixupArgs,
from numba.core.typed_passes import (NopythonTypeInference, AnnotateTypes,
from numba.core.compiler_machinery import FunctionPass, PassManager, register_pass
from numba.experimental import jitclass
import unittest
@skip_unsupported
def test_array_T_issue_3700(self):

    def test_impl(t_obj, X):
        for i in prange(t_obj.T):
            X[i] = i
        return X.sum()
    n = 5
    t_obj = ExampleClass3700(n)
    X1 = np.zeros(t_obj.T)
    X2 = np.zeros(t_obj.T)
    self.assertEqual(njit(test_impl, parallel=True)(t_obj, X1), test_impl(t_obj, X2))