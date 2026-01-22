import ast
from pyflakes import messages as m, checker
from pyflakes.test.harness import TestCase, skip
def test_definedAsStarUnpack(self):
    """Star names in unpack are defined."""
    self.flakes('\n        a, *b = range(10)\n        print(a, b)\n        ')
    self.flakes('\n        *a, b = range(10)\n        print(a, b)\n        ')
    self.flakes('\n        a, *b, c = range(10)\n        print(a, b, c)\n        ')