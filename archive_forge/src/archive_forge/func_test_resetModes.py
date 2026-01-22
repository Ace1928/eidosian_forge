import textwrap
from typing import Optional, Type
from twisted.conch.insults.insults import (
from twisted.internet.protocol import Protocol
from twisted.internet.testing import StringTransport
from twisted.python.compat import iterbytes
from twisted.python.constants import ValueConstant, Values
from twisted.trial import unittest
def test_resetModes(self):
    """
        L{ServerProtocol.resetModes} writes the control sequence
        ending in the L{CSFinalByte.RM}.
        """
    modesToSet = [modes.KAM, modes.IRM, modes.LNM]
    self.protocol.resetModes(modesToSet)
    self.assertEqual(self.transport.value(), self.CSI + b';'.join((b'%d' % (m,) for m in modesToSet)) + CSFinalByte.RM.value)