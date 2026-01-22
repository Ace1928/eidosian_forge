from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_redefinedIfElseFunction(self):
    """
        Test that shadowing a function definition twice in an if
        and else block does not raise a warning.
        """
    self.flakes('\n        if True:\n            def a(): pass\n        else:\n            def a(): pass\n        ')