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
def custom_jit(*args, **kwargs):
    assert 'target' not in kwargs
    assert '_target' not in kwargs
    return njit(*args, _target=CUSTOM_TARGET, **kwargs)