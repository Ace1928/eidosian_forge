from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_listUnpacking(self):
    """
        Don't warn when a variable included in list unpacking is unused.
        """
    self.flakes('\n        def f(tup):\n            [x, y] = tup\n        ')
    self.flakes('\n        def f():\n            [x, y] = [1, 2]\n        ', m.UnusedVariable, m.UnusedVariable)