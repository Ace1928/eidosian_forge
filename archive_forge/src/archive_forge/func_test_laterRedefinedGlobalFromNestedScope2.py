import ast
from pyflakes import messages as m, checker
from pyflakes.test.harness import TestCase, skip
def test_laterRedefinedGlobalFromNestedScope2(self):
    """
        Test that referencing a local name in a nested scope that shadows a
        global declared in an enclosing scope, before it is defined, generates
        a warning.
        """
    self.flakes('\n            a = 1\n            def fun():\n                global a\n                def fun2():\n                    a\n                    a = 2\n                    return a\n        ', m.UndefinedLocal)