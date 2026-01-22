from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_usedInIfBody(self):
    self.flakes('\n        import fu\n        if True: print(fu)\n        ')