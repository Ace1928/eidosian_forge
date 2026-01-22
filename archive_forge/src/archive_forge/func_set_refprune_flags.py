import unittest
import warnings
from contextlib import contextmanager
import numpy as np
import llvmlite.binding as llvm
from numba import njit, types
from numba.core.errors import NumbaInvalidConfigWarning
from numba.core.codegen import _parse_refprune_flags
from numba.tests.support import override_config, TestCase
@contextmanager
def set_refprune_flags(flags):
    with override_config('LLVM_REFPRUNE_FLAGS', flags):
        yield