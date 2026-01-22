import textwrap
from typing import Optional, Type
from twisted.conch.insults.insults import (
from twisted.internet.protocol import Protocol
from twisted.internet.testing import StringTransport
from twisted.python.compat import iterbytes
from twisted.python.constants import ValueConstant, Values
from twisted.trial import unittest
def test_cursorDown(self):
    """
        L{ServerProtocol.cursorDown} writes the control sequence
        ending with L{CSFinalByte.CUD} to its transport.
        """
    self.protocol.cursorDown(1)
    self.assertEqual(self.transport.value(), self.CSI + b'1' + CSFinalByte.CUD.value)