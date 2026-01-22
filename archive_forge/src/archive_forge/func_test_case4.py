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
def test_case4(self):
    """
        Same as case2 but flex_call_fixed() is invoked outside of CUSTOM_TARGET
        target before the switch_target.
        """
    flex_target = self.functions['flex_target']
    flex_call_fixed = self.functions['flex_call_fixed']
    r = flex_call_fixed(123)
    self.assertEqual(r, 123 + 100 + 10)

    @njit
    def foo(x):
        x = flex_call_fixed(x)
        x = flex_target(x)
        return x
    with self.check_retarget_error():
        with self.switch_target():
            foo(123)