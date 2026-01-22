from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_redefinedInClass(self):
    """
        Test that shadowing a global with a class attribute does not produce a
        warning.
        """
    self.flakes('\n        import fu\n        class bar:\n            fu = 1\n        print(fu)\n        ')