from sys import version_info
from pyflakes.test.harness import TestCase, skipIf
def test_defined_in_different_branches(self):
    self.flakes('\n            def f(x):\n                match x:\n                    case 1:\n                        def y(): pass\n                    case _:\n                        def y(): print(1)\n                return y\n        ')