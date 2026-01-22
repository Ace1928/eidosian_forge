import textwrap
from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.test_other import Test as TestOther
from pyflakes.test.test_imports import Test as TestImports
from pyflakes.test.test_undefined_names import Test as TestUndefinedNames
from pyflakes.test.harness import TestCase, skip
def test_nested_class(self):
    """Doctest within nested class are processed."""
    self.flakes("\n        class C:\n            class D:\n                '''\n                    >>> m\n                '''\n                def doctest_stuff(self):\n                    '''\n                        >>> m\n                    '''\n                    return 1\n        ", m.UndefinedName, m.UndefinedName)