import textwrap
from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.test_other import Test as TestOther
from pyflakes.test.test_imports import Test as TestImports
from pyflakes.test.test_undefined_names import Test as TestUndefinedNames
from pyflakes.test.harness import TestCase, skip
def test_offsetInDoctests(self):
    exc = self.flakes('\n\n        def doctest_stuff():\n            """\n                >>> x # line 5\n            """\n\n        ', m.UndefinedName).messages[0]
    self.assertEqual(exc.lineno, 5)
    self.assertEqual(exc.col, 12)