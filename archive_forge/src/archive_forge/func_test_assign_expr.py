from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_assign_expr(self):
    """Test PEP 572 assignment expressions are treated as usage / write."""
    self.flakes('\n        from foo import y\n        print(x := y)\n        print(x)\n        ')