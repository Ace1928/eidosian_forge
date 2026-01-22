from __future__ import unicode_literals
import unittest
from io import StringIO
import string
from .. import Scanning
from ..Symtab import ModuleScope
from ..TreeFragment import StringParseContext
from ..Errors import init_thread
def test_tentatively_scan(self):
    scanner = self.make_scanner()
    with Scanning.tentatively_scan(scanner) as errors:
        while scanner.sy != 'NEWLINE':
            scanner.next()
    self.assertFalse(errors)
    scanner.next()
    self.assertEqual(scanner.systring, 'b0')
    pos = scanner.position()
    with Scanning.tentatively_scan(scanner) as errors:
        while scanner.sy != 'NEWLINE':
            scanner.next()
            if scanner.systring == 'b7':
                scanner.error('Oh no not b7!')
                break
    self.assertTrue(errors)
    self.assertEqual(scanner.systring, 'b0')
    self.assertEqual(scanner.position(), pos)
    scanner.next()
    self.assertEqual(scanner.systring, 'b1')
    scanner.next()
    self.assertEqual(scanner.systring, 'b2')
    with Scanning.tentatively_scan(scanner) as error:
        scanner.error('Something has gone wrong with the current symbol')
    self.assertEqual(scanner.systring, 'b2')
    scanner.next()
    self.assertEqual(scanner.systring, 'b3')
    sy1, systring1 = (scanner.sy, scanner.systring)
    pos1 = scanner.position()
    with Scanning.tentatively_scan(scanner):
        scanner.next()
        sy2, systring2 = (scanner.sy, scanner.systring)
        pos2 = scanner.position()
        with Scanning.tentatively_scan(scanner):
            with Scanning.tentatively_scan(scanner):
                scanner.next()
                scanner.next()
                scanner.error('Ooops')
            self.assertEqual((scanner.sy, scanner.systring), (sy2, systring2))
        self.assertEqual((scanner.sy, scanner.systring), (sy2, systring2))
        scanner.error('eee')
    self.assertEqual((scanner.sy, scanner.systring), (sy1, systring1))
    with Scanning.tentatively_scan(scanner):
        scanner.next()
        scanner.next()
        with Scanning.tentatively_scan(scanner):
            scanner.next()
        scanner.next()
        scanner.error('Oooops')
    self.assertEqual((scanner.sy, scanner.systring), (sy1, systring1))