from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_importedInClass(self):
    """Imports in class scope can be used through self."""
    self.flakes('\n        class c:\n            import i\n            def __init__(self):\n                self.i\n        ')