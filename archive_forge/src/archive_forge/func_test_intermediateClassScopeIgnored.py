import ast
from pyflakes import messages as m, checker
from pyflakes.test.harness import TestCase, skip
def test_intermediateClassScopeIgnored(self):
    """
        If a name defined in an enclosing scope is shadowed by a local variable
        and the name is used locally before it is bound, an unbound local
        warning is emitted, even if there is a class scope between the enclosing
        scope and the local scope.
        """
    self.flakes('\n        def f():\n            x = 1\n            class g:\n                def h(self):\n                    a = x\n                    x = None\n                    print(x, a)\n            print(x)\n        ', m.UndefinedLocal)