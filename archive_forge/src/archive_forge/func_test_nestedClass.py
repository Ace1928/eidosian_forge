import ast
from pyflakes import messages as m, checker
from pyflakes.test.harness import TestCase, skip
def test_nestedClass(self):
    """Nested classes can access enclosing scope."""
    self.flakes('\n        def f(foo):\n            class C:\n                bar = foo\n                def f(self):\n                    return foo\n            return C()\n\n        f(123).f()\n        ')