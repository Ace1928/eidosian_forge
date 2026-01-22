import ast
from pyflakes import messages as m, checker
from pyflakes.test.harness import TestCase, skip
def test_delWhileTestUsage(self):
    """
        Ignore bindings deletion if called inside the body of a while
        statement and name is used inside while's test part.
        """
    self.flakes('\n        def _worker():\n            o = True\n            while o is not True:\n                del o\n                o = False\n        ')