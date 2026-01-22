import ast
from pyflakes import messages as m, checker
from pyflakes.test.harness import TestCase, skip
def test_builtinWindowsError(self):
    """
        C{WindowsError} is sometimes a builtin name, so no warning is emitted
        for using it.
        """
    self.flakes('WindowsError')