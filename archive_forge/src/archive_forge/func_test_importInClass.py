from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_importInClass(self):
    """
        Test that import within class is a locally scoped attribute.
        """
    self.flakes('\n        class bar:\n            import fu\n        ')
    self.flakes('\n        class bar:\n            import fu\n\n        fu\n        ', m.UndefinedName)