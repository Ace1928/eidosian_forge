from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_usedImport(self):
    self.flakes('import fu; print(fu)')
    self.flakes('from baz import fu; print(fu)')
    self.flakes('import fu; del fu')