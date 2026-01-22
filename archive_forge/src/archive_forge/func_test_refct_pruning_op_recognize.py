import math
import os
import platform
import sys
import re
import numpy as np
from numba import njit
from numba.core import types
from numba.core.runtime import (
from numba.core.extending import intrinsic, include_path
from numba.core.typing import signature
from numba.core.imputils import impl_ret_untracked
from llvmlite import ir
import llvmlite.binding as llvm
from numba.core.unsafe.nrt import NRT_get_api
from numba.tests.support import (EnableNRTStatsMixin, TestCase, temp_directory,
from numba.core.registry import cpu_target
import unittest
def test_refct_pruning_op_recognize(self):
    input_ir = self.sample_llvm_ir
    input_lines = list(input_ir.splitlines())
    before_increfs = [ln for ln in input_lines if 'NRT_incref' in ln]
    before_decrefs = [ln for ln in input_lines if 'NRT_decref' in ln]
    output_ir = nrtopt._remove_redundant_nrt_refct(input_ir)
    output_lines = list(output_ir.splitlines())
    after_increfs = [ln for ln in output_lines if 'NRT_incref' in ln]
    after_decrefs = [ln for ln in output_lines if 'NRT_decref' in ln]
    self.assertNotEqual(before_increfs, after_increfs)
    self.assertNotEqual(before_decrefs, after_decrefs)
    pruned_increfs = set(before_increfs) - set(after_increfs)
    pruned_decrefs = set(before_decrefs) - set(after_decrefs)
    combined = pruned_increfs | pruned_decrefs
    self.assertEqual(combined, pruned_increfs ^ pruned_decrefs)
    pruned_lines = '\n'.join(combined)
    for i in [1, 2, 3, 4, 5]:
        gone = '; GONE {}'.format(i)
        self.assertIn(gone, pruned_lines)
    self.assertEqual(len(list(pruned_lines.splitlines())), len(combined))