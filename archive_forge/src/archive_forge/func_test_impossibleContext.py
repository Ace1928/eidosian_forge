import ast
from pyflakes import messages as m, checker
from pyflakes.test.harness import TestCase, skip
def test_impossibleContext(self):
    """
        A Name node with an unrecognized context results in a RuntimeError being
        raised.
        """
    tree = ast.parse('x = 10')
    tree.body[0].targets[0].ctx = object()
    self.assertRaises(RuntimeError, checker.Checker, tree)