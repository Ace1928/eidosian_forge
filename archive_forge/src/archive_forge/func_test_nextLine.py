import textwrap
from typing import Optional, Type
from twisted.conch.insults.insults import (
from twisted.internet.protocol import Protocol
from twisted.internet.testing import StringTransport
from twisted.python.compat import iterbytes
from twisted.python.constants import ValueConstant, Values
from twisted.trial import unittest
def test_nextLine(self):
    """
        L{ServerProtocol.nextLine} writes C{"\r
"} to its transport.
        """
    self.protocol.nextLine()
    self.assertEqual(self.transport.value(), b'\r\n')