import ast
from pyflakes import messages as m, checker
from pyflakes.test.harness import TestCase, skip
def test_definedInListComp(self):
    self.flakes('[a for a in range(10) if a]')