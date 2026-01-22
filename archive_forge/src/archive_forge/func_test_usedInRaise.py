from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_usedInRaise(self):
    self.flakes('\n        import fu\n        raise fu.bar\n        ')