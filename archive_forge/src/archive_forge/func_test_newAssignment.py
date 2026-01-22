from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_newAssignment(self):
    self.flakes('fu = None')