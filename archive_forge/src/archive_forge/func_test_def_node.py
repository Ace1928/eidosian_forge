import os
import tempfile
import unittest
from Cython.Shadow import inline
from Cython.Build.Inline import safe_type
from Cython.TestUtils import CythonTest
def test_def_node(self):
    foo = inline('def foo(x): return x * x', **self._call_kwds)['foo']
    self.assertEqual(foo(7), 49)