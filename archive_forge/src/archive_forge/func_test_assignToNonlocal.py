from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_assignToNonlocal(self):
    """
        Assigning to a nonlocal and then not using that binding is perfectly
        acceptable. Do not mistake it for an unused local variable.
        """
    self.flakes("\n        b = b'0'\n        def a():\n            nonlocal b\n            b = b'1'\n        ")