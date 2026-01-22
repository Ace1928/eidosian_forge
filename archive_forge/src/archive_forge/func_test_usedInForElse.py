from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_usedInForElse(self):
    self.flakes('\n        import fu\n        for bar in range(10):\n            pass\n        else:\n            print(fu)\n        ')