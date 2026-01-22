import ast
from pyflakes import messages as m, checker
from pyflakes.test.harness import TestCase, skip
def test_delWhileNested(self):
    """
        Ignore bindings deletions if node is part of while's test, even when
        del is in a nested block.
        """
    self.flakes('\n        context = None\n        def _worker():\n            o = True\n            while o is not True:\n                while True:\n                    with context():\n                        del o\n                o = False\n        ')