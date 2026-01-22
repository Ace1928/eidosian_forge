import textwrap
from typing import Optional, Type
from twisted.conch.insults.insults import (
from twisted.internet.protocol import Protocol
from twisted.internet.testing import StringTransport
from twisted.python.compat import iterbytes
from twisted.python.constants import ValueConstant, Values
from twisted.trial import unittest
def test_cursorPosition(self):
    """
        L{ServerProtocol.cursorPosition} writes a control sequence
        ending with L{CSFinalByte.CUP} and containing the expected
        coordinates to its transport.
        """
    self.protocol.cursorPosition(0, 0)
    self.assertEqual(self.transport.value(), self.CSI + b'1;1' + CSFinalByte.CUP.value)