import textwrap
from typing import Optional, Type
from twisted.conch.insults.insults import (
from twisted.internet.protocol import Protocol
from twisted.internet.testing import StringTransport
from twisted.python.compat import iterbytes
from twisted.python.constants import ValueConstant, Values
from twisted.trial import unittest
def testHeightAndWidth(self):
    self.parser.dataReceived(b'\x1b#3\x1b#4\x1b#5\x1b#6')
    occs = occurrences(self.proto)
    result = self.assertCall(occs.pop(0), 'doubleHeightLine', (True,))
    self.assertFalse(occurrences(result))
    result = self.assertCall(occs.pop(0), 'doubleHeightLine', (False,))
    self.assertFalse(occurrences(result))
    result = self.assertCall(occs.pop(0), 'singleWidthLine')
    self.assertFalse(occurrences(result))
    result = self.assertCall(occs.pop(0), 'doubleWidthLine')
    self.assertFalse(occurrences(result))
    self.assertFalse(occs)