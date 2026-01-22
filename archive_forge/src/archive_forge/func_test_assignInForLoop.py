from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_assignInForLoop(self):
    """
        Don't warn when a variable in a for loop is assigned to but not used.
        """
    self.flakes('\n        def f():\n            for i in range(10):\n                pass\n        ')