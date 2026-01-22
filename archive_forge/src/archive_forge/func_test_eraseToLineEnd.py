import textwrap
from typing import Optional, Type
from twisted.conch.insults.insults import (
from twisted.internet.protocol import Protocol
from twisted.internet.testing import StringTransport
from twisted.python.compat import iterbytes
from twisted.python.constants import ValueConstant, Values
from twisted.trial import unittest
def test_eraseToLineEnd(self):
    """
        L{ServerProtocol.eraseToLineEnd} writes the control sequence
        sequence ending in L{CSFinalByte.EL} and no parameters,
        forcing the client to default to 0 (from the active present
        position's current location to the end of the line.)
        """
    self.protocol.eraseToLineEnd()
    self.assertEqual(self.transport.value(), self.CSI + CSFinalByte.EL.value)