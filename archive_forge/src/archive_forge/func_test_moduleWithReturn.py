from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_moduleWithReturn(self):
    """
        If a return is used at the module level, a warning is emitted.
        """
    self.flakes('\n        return\n        ', m.ReturnOutsideFunction)