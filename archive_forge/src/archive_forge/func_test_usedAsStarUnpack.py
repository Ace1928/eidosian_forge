import ast
from pyflakes import messages as m, checker
from pyflakes.test.harness import TestCase, skip
def test_usedAsStarUnpack(self):
    """
        Star names in unpack are used if RHS is not a tuple/list literal.
        """
    self.flakes('\n        def f():\n            a, *b = range(10)\n        ')
    self.flakes('\n        def f():\n            (*a, b) = range(10)\n        ')
    self.flakes('\n        def f():\n            [a, *b, c] = range(10)\n        ')