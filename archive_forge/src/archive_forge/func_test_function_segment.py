from pyflakes import messages as m
from pyflakes.checker import (FunctionScope, ClassScope, ModuleScope,
from pyflakes.test.harness import TestCase
def test_function_segment(self):
    self.flakes('\n        def foo():\n            def bar():\n                pass\n        ', is_segment=True)
    self.flakes('\n        def foo():\n            def bar():\n                x = 0\n        ', m.UnusedVariable, is_segment=True)