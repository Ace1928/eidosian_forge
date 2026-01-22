from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_dictComprehension(self):
    """
        Dict comprehensions are properly handled.
        """
    self.flakes('\n        a = {1: x for x in range(10)}\n        ')