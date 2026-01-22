import unittest
import warnings
from contextlib import contextmanager
import numpy as np
import llvmlite.binding as llvm
from numba import njit, types
from numba.core.errors import NumbaInvalidConfigWarning
from numba.core.codegen import _parse_refprune_flags
from numba.tests.support import override_config, TestCase

        Asserts the the func compiled with argument types "argtys" reports
        refop pruning statistics. The **prune_types** kwargs list each kind
        of pruning and whether the stat should be zero (False) or >0 (True).

        Note: The exact statistic varies across platform.

        NOTE: Tests using this `check` method need to run in subprocesses as
        `njit` sets up the module pass manager etc once and the overrides have
        no effect else.
        