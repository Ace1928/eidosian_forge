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
@contextmanager
def check_retarget_error(self):
    with self.assertRaises(errors.NumbaError) as raises:
        yield
    self.assertIn(f'{CUSTOM_TARGET} != cpu', str(raises.exception))