import math
import os
import re
import dis
import numbers
import platform
import sys
import subprocess
import types as pytypes
import warnings
from functools import reduce
import numpy as np
from numpy.random import randn
import operator
from collections import defaultdict, namedtuple
import copy
from itertools import cycle, chain
import subprocess as subp
import numba.parfors.parfor
from numba import (njit, prange, parallel_chunksize,
from numba.core import (types, errors, ir, rewrites,
from numba.extending import (overload_method, register_model,
from numba.core.registry import cpu_target
from numba.core.annotations import type_annotations
from numba.core.ir_utils import (find_callname, guard, build_definitions,
from numba.np.unsafe.ndarray import empty_inferred as unsafe_empty
from numba.core.compiler import (CompilerBase, DefaultPassBuilder)
from numba.core.compiler_machinery import register_pass, AnalysisPass
from numba.core.typed_passes import IRLegalization
from numba.tests.support import (TestCase, captured_stdout, MemoryLeakMixin,
from numba.core.extending import register_jitable
from numba.core.bytecode import _fix_LOAD_GLOBAL_arg
from numba.core import utils
import cmath
import unittest
def test_issue_5098(self):

    class DummyType(types.Opaque):
        pass
    dummy_type = DummyType('my_dummy')
    register_model(DummyType)(models.OpaqueModel)

    class Dummy(object):
        pass

    @typeof_impl.register(Dummy)
    def typeof_Dummy(val, c):
        return dummy_type

    @unbox(DummyType)
    def unbox_index(typ, obj, c):
        return NativeValue(c.context.get_dummy_value())

    @overload_method(DummyType, 'method1', jit_options={'parallel': True})
    def _get_method1(obj, arr, func):

        def _foo(obj, arr, func):

            def baz(a, f):
                c = a.copy()
                c[np.isinf(a)] = np.nan
                return f(c)
            length = len(arr)
            output_arr = np.empty(length, dtype=np.float64)
            for i in prange(length):
                output_arr[i] = baz(arr[i], func)
            for i in prange(length - 1):
                output_arr[i] += baz(arr[i], func)
            return output_arr
        return _foo

    @njit
    def bar(v):
        return v.mean()

    @njit
    def test1(d):
        return d.method1(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), bar)
    save_state = numba.parfors.parfor.sequential_parfor_lowering
    self.assertFalse(save_state)
    try:
        test1(Dummy())
        self.assertFalse(numba.parfors.parfor.sequential_parfor_lowering)
    finally:
        numba.parfors.parfor.sequential_parfor_lowering = save_state