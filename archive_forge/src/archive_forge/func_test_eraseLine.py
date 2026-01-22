import textwrap
from typing import Optional, Type
from twisted.conch.insults.insults import (
from twisted.internet.protocol import Protocol
from twisted.internet.testing import StringTransport
from twisted.python.compat import iterbytes
from twisted.python.constants import ValueConstant, Values
from twisted.trial import unittest
def test_eraseLine(self):
    """
        L{ServerProtocol.eraseLine} writes the control
        sequence sequence ending in L{CSFinalByte.EL} and a parameter
        of 2 (the entire line.)
        """
    self.protocol.eraseLine()
    self.assertEqual(self.transport.value(), self.CSI + b'2' + CSFinalByte.EL.value)