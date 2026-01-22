from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_usedInFor(self):
    self.flakes('\n        import fu\n        for bar in range(9):\n            print(fu)\n        ')