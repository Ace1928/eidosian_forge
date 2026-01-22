from __future__ import annotations
import contextlib
import ctypes
import functools
from ctypes import (
from ctypes.util import find_library
@functools.singledispatch
def create_cf(ob):
    return ob