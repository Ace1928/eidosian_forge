import ast
from pyflakes import messages as m, checker
from pyflakes.test.harness import TestCase, skip
def test_magicModuleInClassScope(self):
    """
        Use of the C{__module__} magic builtin should not emit an undefined
        name warning if used in class scope.
        """
    self.flakes('__module__', m.UndefinedName)
    self.flakes('\n        class Foo:\n            __module__\n        ')
    self.flakes('\n        class Foo:\n            def bar(self):\n                __module__\n        ', m.UndefinedName)