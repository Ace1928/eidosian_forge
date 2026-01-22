from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
@skipIf(version_info < (3, 11), 'new in Python 3.11')
def test_exception_unused_in_except_star(self):
    self.flakes('\n            try:\n                pass\n            except* OSError as e:\n                pass\n        ', m.UnusedVariable)