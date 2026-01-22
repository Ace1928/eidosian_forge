import textwrap
from typing import Optional, Type
from twisted.conch.insults.insults import (
from twisted.internet.protocol import Protocol
from twisted.internet.testing import StringTransport
from twisted.python.compat import iterbytes
from twisted.python.constants import ValueConstant, Values
from twisted.trial import unittest
def test_insertLine(self):
    """
        L{ServerProtocol.insertLine} writes the control sequence
        containing the number of lines to insert and ending in
        L{CSFinalByte.IL}
        """
    self.protocol.insertLine(5)
    self.assertEqual(self.transport.value(), self.CSI + b'5' + CSFinalByte.IL.value)