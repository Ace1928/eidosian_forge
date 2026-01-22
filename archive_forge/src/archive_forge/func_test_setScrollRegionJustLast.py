import textwrap
from typing import Optional, Type
from twisted.conch.insults.insults import (
from twisted.internet.protocol import Protocol
from twisted.internet.testing import StringTransport
from twisted.python.compat import iterbytes
from twisted.python.constants import ValueConstant, Values
from twisted.trial import unittest
def test_setScrollRegionJustLast(self):
    """
        With just a value for its C{last} argument,
        L{ServerProtocol.setScrollRegion} writes a control sequence with
        a parameter separator, that parameter, and finally a C{b'r'}.
        """
    self.protocol.setScrollRegion(last=1)
    self.assertEqual(self.transport.value(), self.CSI + b';1' + b'r')