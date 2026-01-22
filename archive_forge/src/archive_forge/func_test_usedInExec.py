from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_usedInExec(self):
    exec_stmt = 'exec("print(1)", fu.bar)'
    self.flakes('import fu; %s' % exec_stmt)