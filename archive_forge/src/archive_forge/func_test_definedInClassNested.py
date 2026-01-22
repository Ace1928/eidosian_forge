import ast
from pyflakes import messages as m, checker
from pyflakes.test.harness import TestCase, skip
def test_definedInClassNested(self):
    """Defined name for nested generator expressions in a class."""
    self.flakes('\n        class A:\n            T = range(10)\n\n            Z = (x for x in (a for a in T))\n        ')