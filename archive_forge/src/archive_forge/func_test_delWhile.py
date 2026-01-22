import ast
from pyflakes import messages as m, checker
from pyflakes.test.harness import TestCase, skip
def test_delWhile(self):
    """
        Ignore bindings deletion if called inside the body of a while
        statement.
        """
    self.flakes("\n        def test():\n            foo = 'bar'\n            while False:\n                del foo\n            assert(foo)\n        ")