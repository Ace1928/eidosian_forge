from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skipIf
def test_unused_annotation(self):
    self.flakes('\n        x: int\n        class Cls:\n            y: int\n        ')
    self.flakes('\n        def f():\n            x: int\n        ', m.UnusedAnnotation)
    self.flakes('\n        def f():\n            x: int\n            x = 3\n        ', m.UnusedVariable)