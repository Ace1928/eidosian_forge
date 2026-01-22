from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_usedInAttributeAssign(self):
    self.flakes('import fu; fu.bar = 1')