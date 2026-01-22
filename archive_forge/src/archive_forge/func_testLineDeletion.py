import textwrap
from typing import Optional, Type
from twisted.conch.insults.insults import (
from twisted.internet.protocol import Protocol
from twisted.internet.testing import StringTransport
from twisted.python.compat import iterbytes
from twisted.python.constants import ValueConstant, Values
from twisted.trial import unittest
def testLineDeletion(self):
    self.parser.dataReceived(b'\x1b[M\x1b[3M')
    occs = occurrences(self.proto)
    for arg in (1, 3):
        result = self.assertCall(occs.pop(0), 'deleteLine', (arg,))
        self.assertFalse(occurrences(result))
    self.assertFalse(occs)