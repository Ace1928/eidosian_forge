import warnings
import base64
import ctypes
import pickle
import re
import subprocess
import sys
import weakref
import llvmlite.binding as ll
import unittest
from numba import njit
from numba.core.codegen import JITCPUCodegen
from numba.core.compiler_lock import global_compiler_lock
from numba.tests.support import TestCase
def test_unserialize_other_process_bitcode(self):
    library = self.compile_module(asm_sum_outer, asm_sum_inner)
    state = library.serialize_using_bitcode()
    self._check_unserialize_other_process(state)