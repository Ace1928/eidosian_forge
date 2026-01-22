from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_formatstring(self):
    self.flakes("\n        hi = 'hi'\n        mom = 'mom'\n        f'{hi} {mom}'\n        ")