import numpy as np
import numpy
import unittest
from numba import njit, jit
from numba.core.errors import TypingError, UnsupportedError
from numba.core import ir
from numba.tests.support import TestCase, IRPreservingTestPipeline
def test_closure_renaming_scheme(self):

    @njit(pipeline_class=IRPreservingTestPipeline)
    def foo(a, b):

        def bar(z):
            x = 5
            y = 10
            return x + y + z
        return (bar(a), bar(b))
    self.assertEqual(foo(10, 20), (25, 35))
    func_ir = foo.overloads[foo.signatures[0]].metadata['preserved_ir']
    store = []
    for blk in func_ir.blocks.values():
        for stmt in blk.body:
            if isinstance(stmt, ir.Assign):
                if isinstance(stmt.value, ir.Const):
                    if stmt.value.value == 5:
                        store.append(stmt)
    self.assertEqual(len(store), 2)
    for i in store:
        name = i.target.name
        regex = 'closure__locals__bar_v[0-9]+.x'
        self.assertRegex(name, regex)