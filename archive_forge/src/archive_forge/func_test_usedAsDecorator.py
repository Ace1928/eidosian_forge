from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_usedAsDecorator(self):
    """
        Using a global name in a decorator statement results in no warnings,
        but using an undefined name in a decorator statement results in an
        undefined name warning.
        """
    self.flakes('\n        from interior import decorate\n        @decorate\n        def f():\n            return "hello"\n        ')
    self.flakes('\n        from interior import decorate\n        @decorate(\'value\')\n        def f():\n            return "hello"\n        ')
    self.flakes('\n        @decorate\n        def f():\n            return "hello"\n        ', m.UndefinedName)