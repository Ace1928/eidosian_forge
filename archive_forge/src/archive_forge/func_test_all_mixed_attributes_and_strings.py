from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_all_mixed_attributes_and_strings(self):
    self.flakes("\n        from foo import bar\n        from foo import baz\n        __all__ = ['bar', baz.__name__]\n        ")