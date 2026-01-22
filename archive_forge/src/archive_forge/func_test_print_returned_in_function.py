from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_print_returned_in_function(self):
    self.flakes('\n        from __future__ import print_function\n        def a():\n            return print\n        ')