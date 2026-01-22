import unittest
from numba.tests.support import TestCase
import ctypes
import operator
from functools import cached_property
import numpy as np
from numba import njit, types
from numba.extending import overload, intrinsic, overload_classmethod
from numba.core.target_extension import (
from numba.core import utils, fastmathpass, errors
from numba.core.dispatcher import Dispatcher
from numba.core.descriptors import TargetDescriptor
from numba.core import cpu, typing, cgutils
from numba.core.base import BaseContext
from numba.core.compiler_lock import global_compiler_lock
from numba.core import callconv
from numba.core.codegen import CPUCodegen, JITCodeLibrary
from numba.core.callwrapper import PyCallWrapper
from numba.core.imputils import RegistryLoader, Registry
from numba import _dynfunc
import llvmlite.binding as ll
from llvmlite import ir as llir
from numba.core.runtime import rtsys
from numba.core import compiler
from numba.core.compiler import CompilerBase, DefaultPassBuilder
from numba.core.compiler_machinery import FunctionPass, register_pass
from numba.core.typed_passes import PreLowerStripPhis
def test_no_specialisation_found(self):

    def my_func(x):
        pass

    @overload(my_func, target='cuda')
    def ol_my_func_cuda(x):
        return lambda x: None

    @djit(nopython=True)
    def dpu_foo():
        my_func(1)
    accept = (errors.UnsupportedError, errors.TypingError)
    with self.assertRaises(accept) as raises:
        dpu_foo()
    msgs = ['Function resolution cannot find any matches for function', 'test_no_specialisation_found.<locals>.my_func', 'for the current target:', "'numba.tests.test_target_extension.DPU'"]
    for msg in msgs:
        self.assertIn(msg, str(raises.exception))