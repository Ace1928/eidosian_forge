import textwrap
from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.test_other import Test as TestOther
from pyflakes.test.test_imports import Test as TestImports
from pyflakes.test.test_undefined_names import Test as TestUndefinedNames
from pyflakes.test.harness import TestCase, skip
def test_globalUnderscoreInDoctest(self):
    self.flakes("\n        from gettext import ugettext as _\n\n        def doctest_stuff():\n            '''\n                >>> pass\n            '''\n        ", m.UnusedImport)