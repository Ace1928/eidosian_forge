from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_unusedVariableAsLocals(self):
    """
        Using locals() it is perfectly valid to have unused variables
        """
    self.flakes('\n        def a():\n            b = 1\n            return locals()\n        ')