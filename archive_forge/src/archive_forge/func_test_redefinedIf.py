from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_redefinedIf(self):
    """
        Test that importing a module twice within an if
        block does raise a warning.
        """
    self.flakes('\n        i = 2\n        if i==1:\n            import os\n            import os\n        os.path', m.RedefinedWhileUnused)