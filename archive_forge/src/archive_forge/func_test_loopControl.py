from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_loopControl(self):
    """
        break and continue statements are supported.
        """
    self.flakes('\n        for x in [1, 2]:\n            break\n        ')
    self.flakes('\n        for x in [1, 2]:\n            continue\n        ')