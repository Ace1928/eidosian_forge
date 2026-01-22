import ast
from pyflakes import messages as m, checker
from pyflakes.test.harness import TestCase, skip
def test_functionsNeedGlobalScope(self):
    self.flakes('\n        class a:\n            def b():\n                fu\n        fu = 1\n        ')