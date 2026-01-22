import os
import platform
import re
import textwrap
import warnings
import numpy as np
from numba.tests.support import (TestCase, override_config, override_env_config,
from numba import jit, njit
from numba.core import types, compiler, utils
from numba.core.errors import NumbaPerformanceWarning
from numba import prange
from numba.experimental import jitclass
import unittest
@skip_parfors_unsupported
def test_array_debug_opt_stats(self):
    """
        Test that NUMBA_DEBUG_ARRAY_OPT_STATS produces valid output
        """
    njit((types.int64,), parallel=True)(supported_parfor)
    with override_env_config('NUMBA_DEBUG_ARRAY_OPT_STATS', '1'):
        with captured_stdout() as out:
            njit((types.int64,), parallel=True)(supported_parfor)
        output = out.getvalue().split('\n')
        parallel_loop_output = [x for x in output if 'is produced from pattern' in x]
        fuse_output = [x for x in output if 'is fused into' in x]
        after_fusion_output = [x for x in output if 'After fusion, function' in x]
        parfor_state = int(re.compile('#([0-9]+)').search(parallel_loop_output[0]).group(1))
        bounds = range(parfor_state, parfor_state + len(parallel_loop_output))
        pattern = ("('ones function', 'NumPy mapping')", ('prange', 'user', ''))
        fmt = "Parallel for-loop #{} is produced from pattern '{}' at"
        for i, trials, lpattern in zip(bounds, parallel_loop_output, pattern):
            to_match = fmt.format(i, lpattern)
            self.assertIn(to_match, trials)
        pattern = (parfor_state + 1, parfor_state + 0)
        fmt = 'Parallel for-loop #{} is fused into for-loop #{}.'
        for trials in fuse_output:
            to_match = fmt.format(*pattern)
            self.assertIn(to_match, trials)
        pattern = (supported_parfor.__name__, 1, set([parfor_state]))
        fmt = 'After fusion, function {} has {} parallel for-loop(s) #{}.'
        for trials in after_fusion_output:
            to_match = fmt.format(*pattern)
            self.assertIn(to_match, trials)