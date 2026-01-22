import textwrap
from typing import Optional, Type
from twisted.conch.insults.insults import (
from twisted.internet.protocol import Protocol
from twisted.internet.testing import StringTransport
from twisted.python.compat import iterbytes
from twisted.python.constants import ValueConstant, Values
from twisted.trial import unittest
def test_setModes(self):
    """
        L{ServerProtocol.setModes} writes a control sequence
        containing the requested modes and ending in the
        L{CSFinalByte.SM}.
        """
    modesToSet = [modes.KAM, modes.IRM, modes.LNM]
    self.protocol.setModes(modesToSet)
    self.assertEqual(self.transport.value(), self.CSI + b';'.join((b'%d' % (m,) for m in modesToSet)) + CSFinalByte.SM.value)