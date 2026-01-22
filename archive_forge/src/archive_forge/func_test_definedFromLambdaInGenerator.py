import ast
from pyflakes import messages as m, checker
from pyflakes.test.harness import TestCase, skip
def test_definedFromLambdaInGenerator(self):
    """
        Defined name referenced from a lambda function within a generator
        expression.
        """
    self.flakes('\n        any(lambda: id(x) for x in range(10))\n        ')