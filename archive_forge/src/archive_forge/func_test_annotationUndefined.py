import ast
from pyflakes import messages as m, checker
from pyflakes.test.harness import TestCase, skip
def test_annotationUndefined(self):
    """Undefined annotations."""
    self.flakes('\n        from abc import note1, note2, note3, note4, note5\n        def func(a: note1, *args: note2,\n                 b: note3=12, **kw: note4) -> note5: pass\n        ')
    self.flakes('\n        def func():\n            d = e = 42\n            def func(a: {1, d}) -> (lambda c: e): pass\n        ')