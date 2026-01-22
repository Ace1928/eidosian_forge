import textwrap
from typing import Optional, Type
from twisted.conch.insults.insults import (
from twisted.internet.protocol import Protocol
from twisted.internet.testing import StringTransport
from twisted.python.compat import iterbytes
from twisted.python.constants import ValueConstant, Values
from twisted.trial import unittest
def testCursorPosition(self):
    methods(self.proto)['reportCursorPosition'] = (6, 7)
    self.parser.dataReceived(b'\x1b[6n')
    self.assertEqual(self.transport.value(), b'\x1b[7;8R')
    occs = occurrences(self.proto)
    result = self.assertCall(occs.pop(0), 'reportCursorPosition')
    self.assertEqual(result, (6, 7))