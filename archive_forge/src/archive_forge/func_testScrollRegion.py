import textwrap
from typing import Optional, Type
from twisted.conch.insults.insults import (
from twisted.internet.protocol import Protocol
from twisted.internet.testing import StringTransport
from twisted.python.compat import iterbytes
from twisted.python.constants import ValueConstant, Values
from twisted.trial import unittest
def testScrollRegion(self):
    self.parser.dataReceived(b'\x1b[5;22r\x1b[r')
    occs = occurrences(self.proto)
    result = self.assertCall(occs.pop(0), 'setScrollRegion', (5, 22))
    self.assertFalse(occurrences(result))
    result = self.assertCall(occs.pop(0), 'setScrollRegion', (None, None))
    self.assertFalse(occurrences(result))
    self.assertFalse(occs)