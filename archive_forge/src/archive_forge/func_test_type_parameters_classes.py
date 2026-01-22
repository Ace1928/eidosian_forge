from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skipIf
@skipIf(version_info < (3, 12), 'new in Python 3.12')
def test_type_parameters_classes(self):
    self.flakes('\n            class C[T](list[T]): pass\n\n            class UsesForward[T: Forward](list[T]): pass\n\n            class Forward: pass\n\n            class WithinBody[T](list[T]):\n                t = T\n        ')