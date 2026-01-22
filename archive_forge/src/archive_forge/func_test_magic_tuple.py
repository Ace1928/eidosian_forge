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
def test_magic_tuple(self):
    tup = self.codegen.magic_tuple()
    pickle.dumps(tup)
    cg2 = JITCPUCodegen('xxx')
    self.assertEqual(cg2.magic_tuple(), tup)