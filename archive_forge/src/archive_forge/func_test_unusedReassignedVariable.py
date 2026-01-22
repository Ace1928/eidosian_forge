from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
@skip("todo: Difficult because it doesn't apply in the context of a loop")
def test_unusedReassignedVariable(self):
    """
        Shadowing a used variable can still raise an UnusedVariable warning.
        """
    self.flakes('\n        def a():\n            b = 1\n            b.foo()\n            b = 2\n        ', m.UnusedVariable)