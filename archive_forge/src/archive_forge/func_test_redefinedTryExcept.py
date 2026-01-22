from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_redefinedTryExcept(self):
    """
        Test that importing a module twice in a try
        and except block does not raise a warning.
        """
    self.flakes('\n        try:\n            import os\n        except:\n            import os\n        os.path')