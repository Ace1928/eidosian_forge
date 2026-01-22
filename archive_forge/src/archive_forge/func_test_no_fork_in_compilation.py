import copy
import warnings
import numpy as np
import numba
from numba.core.transforms import find_setupwiths, with_lifting
from numba.core.withcontexts import bypass_context, call_context, objmode_context
from numba.core.bytecode import FunctionIdentity, ByteCode
from numba.core.interpreter import Interpreter
from numba.core import errors
from numba.core.registry import cpu_target
from numba.core.compiler import compile_ir, DEFAULT_FLAGS
from numba import njit, typeof, objmode, types
from numba.core.extending import overload
from numba.tests.support import (MemoryLeak, TestCase, captured_stdout,
from numba.core.utils import PYVERSION
from numba.experimental import jitclass
import unittest
@linux_only
@TestCase.run_test_in_subprocess
def test_no_fork_in_compilation(self):
    if not strace_supported():
        self.skipTest('strace support missing')

    def force_compile():

        @njit('void()')
        def f():
            with numba.objmode():
                pass
    syscalls = ['fork', 'clone', 'execve']
    strace_data = strace(force_compile, syscalls)
    self.assertFalse(strace_data)