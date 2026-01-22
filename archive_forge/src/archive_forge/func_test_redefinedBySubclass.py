from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_redefinedBySubclass(self):
    """
        If an imported name is redefined by a class statement which also uses
        that name in the bases list, no warning is emitted.
        """
    self.flakes('\n        from fu import bar\n        class bar(bar):\n            pass\n        ')