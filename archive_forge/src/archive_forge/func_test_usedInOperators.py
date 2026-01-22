from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_usedInOperators(self):
    self.flakes('import fu; 3 + fu.bar')
    self.flakes('import fu; 3 % fu.bar')
    self.flakes('import fu; 3 - fu.bar')
    self.flakes('import fu; 3 * fu.bar')
    self.flakes('import fu; 3 ** fu.bar')
    self.flakes('import fu; 3 / fu.bar')
    self.flakes('import fu; 3 // fu.bar')
    self.flakes('import fu; -fu.bar')
    self.flakes('import fu; ~fu.bar')
    self.flakes('import fu; 1 == fu.bar')
    self.flakes('import fu; 1 | fu.bar')
    self.flakes('import fu; 1 & fu.bar')
    self.flakes('import fu; 1 ^ fu.bar')
    self.flakes('import fu; 1 >> fu.bar')
    self.flakes('import fu; 1 << fu.bar')