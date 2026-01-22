from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
@skip('todo')
def test_importingForImportError(self):
    self.flakes('\n        try:\n            import fu\n        except ImportError:\n            pass\n        ')