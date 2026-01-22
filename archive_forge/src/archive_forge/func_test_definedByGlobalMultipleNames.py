import ast
from pyflakes import messages as m, checker
from pyflakes.test.harness import TestCase, skip
def test_definedByGlobalMultipleNames(self):
    """
        "global" can accept multiple names.
        """
    self.flakes('\n        def a(): global fu, bar; fu = 1; bar = 2\n        def b(): fu; bar\n        ')