import textwrap
from typing import Optional, Type
from twisted.conch.insults.insults import (
from twisted.internet.protocol import Protocol
from twisted.internet.testing import StringTransport
from twisted.python.compat import iterbytes
from twisted.python.constants import ValueConstant, Values
from twisted.trial import unittest
def test_deleteCharacter(self):
    """
        L{ServerProtocol.deleteCharacter} writes the control sequence
        containing the number of characters to delete and ending in
        L{CSFinalByte.DCH}
        """
    self.protocol.deleteCharacter(4)
    self.assertEqual(self.transport.value(), self.CSI + b'4' + CSFinalByte.DCH.value)