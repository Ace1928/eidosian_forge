from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_assignRHSFirst(self):
    self.flakes('import fu; fu = fu')
    self.flakes('import fu; fu, bar = fu')
    self.flakes('import fu; [fu, bar] = fu')
    self.flakes('import fu; fu += fu')