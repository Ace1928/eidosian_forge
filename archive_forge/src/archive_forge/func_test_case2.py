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
def test_case2(self):
    """
        The non-nested call into fixed_target should raise error.
        """
    fixed_target = self.functions['fixed_target']
    flex_target = self.functions['flex_target']

    @njit
    def foo(x):
        x = fixed_target(x)
        x = flex_target(x)
        return x
    with self.check_retarget_error():
        with self.switch_target():
            foo(123)