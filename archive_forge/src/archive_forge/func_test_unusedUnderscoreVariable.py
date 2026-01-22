from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_unusedUnderscoreVariable(self):
    """
        Don't warn when the magic "_" (underscore) variable is unused.
        See issue #202.
        """
    self.flakes('\n        def a(unused_param):\n            _ = unused_param\n        ')