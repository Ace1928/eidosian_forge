from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skipIf
def test_unassigned_annotation_is_undefined(self):
    self.flakes('\n        name: str\n        print(name)\n        ', m.UndefinedName)