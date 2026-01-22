import unittest
import warnings
from contextlib import contextmanager
import numpy as np
import llvmlite.binding as llvm
from numba import njit, types
from numba.core.errors import NumbaInvalidConfigWarning
from numba.core.codegen import _parse_refprune_flags
from numba.tests.support import override_config, TestCase
def test_some_flags(self):
    with set_refprune_flags('per_bb, fanout'):
        optval = _parse_refprune_flags()
        enumcls = llvm.RefPruneSubpasses
        self.assertEqual(optval, enumcls.PER_BB | enumcls.FANOUT)