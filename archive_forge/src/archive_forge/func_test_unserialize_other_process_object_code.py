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
def test_unserialize_other_process_object_code(self):
    library = self.compile_module(asm_sum_outer, asm_sum_inner)
    library.enable_object_caching()
    state = library.serialize_using_object_code()
    self._check_unserialize_other_process(state)