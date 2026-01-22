from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_invalid_print_when_imported_from_future(self):
    exc = self.flakes('\n        from __future__ import print_function\n        import sys\n        print >>sys.stderr, "Hello"\n        ', m.InvalidPrintSyntax).messages[0]
    self.assertEqual(exc.lineno, 4)
    self.assertEqual(exc.col, 0)