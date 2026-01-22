from __future__ import unicode_literals
import unittest
from io import StringIO
import string
from .. import Scanning
from ..Symtab import ModuleScope
from ..TreeFragment import StringParseContext
from ..Errors import init_thread
def test_put_back_positions(self):
    scanner = self.make_scanner()
    self.assertEqual(scanner.sy, 'IDENT')
    self.assertEqual(scanner.systring, 'a0')
    scanner.next()
    self.assertEqual(scanner.sy, 'IDENT')
    self.assertEqual(scanner.systring, 'a1')
    a1pos = scanner.position()
    self.assertEqual(a1pos[1:], (1, 3))
    a2peek = scanner.peek()
    self.assertEqual(a1pos, scanner.position())
    scanner.next()
    self.assertEqual(a2peek, (scanner.sy, scanner.systring))
    while scanner.sy != 'NEWLINE':
        scanner.next()
    line_sy = []
    line_systring = []
    line_pos = []
    scanner.next()
    while scanner.sy != 'NEWLINE':
        line_sy.append(scanner.sy)
        line_systring.append(scanner.systring)
        line_pos.append(scanner.position())
        scanner.next()
    for sy, systring, pos in zip(line_sy[::-1], line_systring[::-1], line_pos[::-1]):
        scanner.put_back(sy, systring, pos)
    n = 0
    while scanner.sy != 'NEWLINE':
        self.assertEqual(scanner.sy, line_sy[n])
        self.assertEqual(scanner.systring, line_systring[n])
        self.assertEqual(scanner.position(), line_pos[n])
        scanner.next()
        n += 1
    self.assertEqual(n, len(line_pos))