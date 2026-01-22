from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skipIf
def test_annotating_an_import(self):
    self.flakes('\n            from a import b, c\n            b: c\n            print(b)\n        ')