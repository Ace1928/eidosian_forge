from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_packageImport(self):
    """
        If a dotted name is imported and used, no warning is reported.
        """
    self.flakes('\n        import fu.bar\n        fu.bar\n        ')