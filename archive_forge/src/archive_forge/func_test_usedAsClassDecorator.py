from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_usedAsClassDecorator(self):
    """
        Using an imported name as a class decorator results in no warnings,
        but using an undefined name as a class decorator results in an
        undefined name warning.
        """
    self.flakes('\n        from interior import decorate\n        @decorate\n        class foo:\n            pass\n        ')
    self.flakes('\n        from interior import decorate\n        @decorate("foo")\n        class bar:\n            pass\n        ')
    self.flakes('\n        @decorate\n        class foo:\n            pass\n        ', m.UndefinedName)