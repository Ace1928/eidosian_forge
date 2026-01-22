import unittest
from contextlib import contextmanager
from functools import cached_property
from numba import njit
from numba.core import errors, cpu, typing
from numba.core.descriptors import TargetDescriptor
from numba.core.dispatcher import TargetConfigurationStack
from numba.core.retarget import BasicRetarget
from numba.core.extending import overload
from numba.core.target_extension import (
def test_case1(self):
    flex_target = self.functions['flex_target']

    @njit
    def foo(x):
        x = flex_target(x)
        return x
    with self.switch_target():
        r = foo(123)
    self.assertEqual(r, 123 + 1000)
    self.check_non_empty_cache()