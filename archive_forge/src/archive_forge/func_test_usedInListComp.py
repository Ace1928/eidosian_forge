from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_usedInListComp(self):
    self.flakes('import fu; [fu for _ in range(1)]')
    self.flakes('import fu; [1 for _ in range(1) if fu]')