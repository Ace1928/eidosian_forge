import ast
from pyflakes import messages as m, checker
from pyflakes.test.harness import TestCase, skip
def test_badNestedClass(self):
    """Free variables in nested classes must bind at class creation."""
    self.flakes('\n        def f():\n            class C:\n                bar = foo\n            foo = 456\n            return foo\n        f()\n        ', m.UndefinedName)