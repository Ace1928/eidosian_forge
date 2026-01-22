from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_moduleWithYieldFrom(self):
    """
        If a yield from is used at the module level, a warning is emitted.
        """
    self.flakes('\n        yield from range(10)\n        ', m.YieldOutsideFunction)