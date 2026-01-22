from collections import namedtuple
import inspect
import re
import numpy as np
import math
from textwrap import dedent
import unittest
import warnings
from numba.tests.support import (TestCase, override_config,
from numba import jit, njit
from numba.core import types
from numba.core.datamodel import default_manager
from numba.core.errors import NumbaDebugInfoWarning
import llvmlite.binding as llvm
def test_llvm_inliner_flag_conflict(self):

    @njit(forceinline=True)
    def bar(x):
        return math.sin(x)

    @njit(forceinline=False)
    def baz(x):
        return math.cos(x)

    @njit
    def foo(x):
        a = bar(x)
        b = baz(x)
        return (a, b)
    with override_config('DEBUGINFO_DEFAULT', 1):
        result = foo(np.pi)
    self.assertPreciseEqual(result, foo.py_func(np.pi))
    full_ir = foo.inspect_llvm(foo.signatures[0])
    module = llvm.parse_assembly(full_ir)
    name = foo.overloads[foo.signatures[0]].fndesc.mangled_name
    funcs = [x for x in module.functions if x.name == name]
    self.assertEqual(len(funcs), 1)
    func = funcs[0]
    f_names = []
    for blk in func.blocks:
        for stmt in blk.instructions:
            if stmt.opcode == 'call':
                f_names.append(str(stmt).strip())
    found_sin = False
    found_baz = False
    baz_name = baz.overloads[baz.signatures[0]].fndesc.mangled_name
    for x in f_names:
        if not found_sin and re.match('.*llvm.sin.f64.*', x):
            found_sin = True
        if not found_baz and re.match(f'.*{baz_name}.*', x):
            found_baz = True
    self.assertTrue(found_sin)
    self.assertTrue(found_baz)