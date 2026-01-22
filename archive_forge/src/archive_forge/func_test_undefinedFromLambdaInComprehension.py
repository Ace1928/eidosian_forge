import ast
from pyflakes import messages as m, checker
from pyflakes.test.harness import TestCase, skip
def test_undefinedFromLambdaInComprehension(self):
    """
        Undefined name referenced from a lambda function within a generator
        expression.
        """
    self.flakes('\n        any(lambda: id(y) for x in range(10))\n        ', m.UndefinedName)