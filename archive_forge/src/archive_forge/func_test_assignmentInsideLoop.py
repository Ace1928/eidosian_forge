from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_assignmentInsideLoop(self):
    """
        Don't warn when a variable assignment occurs lexically after its use.
        """
    self.flakes('\n        def f():\n            x = None\n            for i in range(10):\n                if i > 2:\n                    return x\n                x = i * 2\n        ')