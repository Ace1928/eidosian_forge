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
def test_stack(m, n):
    a = np.ones(m)
    b = np.ones(n)
    c = np.stack((a, b))
    d = np.ones((m, n))
    e = np.ones((m, n))
    f = np.stack((d, e))
    g = np.stack((d, e), axis=0)
    h = np.stack((d, e), axis=1)
    i = np.stack((d, e), axis=2)
    j = np.stack((d, e), axis=-1)