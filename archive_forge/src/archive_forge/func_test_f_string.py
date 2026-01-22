from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_f_string(self):
    """Test PEP 498 f-strings are treated as a usage."""
    self.flakes("\n        baz = 0\n        print(f'{4*baz}')\n        ")