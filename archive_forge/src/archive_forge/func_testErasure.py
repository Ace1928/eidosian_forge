import textwrap
from typing import Optional, Type
from twisted.conch.insults.insults import (
from twisted.internet.protocol import Protocol
from twisted.internet.testing import StringTransport
from twisted.python.compat import iterbytes
from twisted.python.constants import ValueConstant, Values
from twisted.trial import unittest
def testErasure(self):
    self.parser.dataReceived(b'\x1b[K\x1b[1K\x1b[2K\x1b[J\x1b[1J\x1b[2J\x1b[3P')
    occs = occurrences(self.proto)
    for meth in ('eraseToLineEnd', 'eraseToLineBeginning', 'eraseLine', 'eraseToDisplayEnd', 'eraseToDisplayBeginning', 'eraseDisplay'):
        result = self.assertCall(occs.pop(0), meth)
        self.assertFalse(occurrences(result))
    result = self.assertCall(occs.pop(0), 'deleteCharacter', (3,))
    self.assertFalse(occurrences(result))
    self.assertFalse(occs)