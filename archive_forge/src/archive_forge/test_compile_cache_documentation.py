import unittest
from contextlib import contextmanager
from llvmlite import ir
from numba.core import types, typing, callconv, cpu, cgutils
from numba.core.registry import cpu_target

        Caching must not mix up different error models.
        