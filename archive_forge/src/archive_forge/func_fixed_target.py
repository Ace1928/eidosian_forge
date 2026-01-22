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
@njit(_target='cpu')
def fixed_target(x):
    """
            This has a fixed target to "cpu".
            Cannot be used in CUSTOM_TARGET target.
            """
    return x + 10