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
def get_gufunc_asm(self, func, schedule_type, *args, **kwargs):
    fastmath = kwargs.pop('fastmath', False)
    cpu_name = kwargs.pop('cpu_name', 'skylake-avx512')
    assertions = kwargs.pop('assertions', True)
    cpu_features = kwargs.pop('cpu_features', '-prefer-256-bit')
    env_opts = {'NUMBA_CPU_NAME': cpu_name, 'NUMBA_CPU_FEATURES': cpu_features}
    overrides = []
    for k, v in env_opts.items():
        overrides.append(override_env_config(k, v))
    with overrides[0], overrides[1]:
        sig = tuple([numba.typeof(x) for x in args])
        pfunc_vectorizable = self.generate_prange_func(func, None)
        if fastmath == True:
            cres = self.compile_parallel_fastmath(pfunc_vectorizable, sig)
        else:
            cres = self.compile_parallel(pfunc_vectorizable, sig)
        asm = self._get_gufunc_asm(cres)
        if assertions:
            schedty = re.compile('call\\s+\\w+\\*\\s+@do_scheduling_(\\w+)\\(')
            matches = schedty.findall(cres.library.get_llvm_str())
            self.assertGreaterEqual(len(matches), 1)
            self.assertEqual(matches[0], schedule_type)
            self.assertNotEqual(asm, {})
        return asm