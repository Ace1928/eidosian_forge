from pyflakes import messages as m
from pyflakes.checker import (FunctionScope, ClassScope, ModuleScope,
from pyflakes.test.harness import TestCase
def test_class_segment(self):
    self.flakes('\n        class Foo:\n            class Bar:\n                pass\n        ', is_segment=True)
    self.flakes('\n        class Foo:\n            def bar():\n                x = 0\n        ', m.UnusedVariable, is_segment=True)