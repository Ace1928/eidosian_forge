import sys
import traceback
from typing import Optional
from twisted.conch import manhole
from twisted.conch.insults import insults
from twisted.conch.test.test_recvline import (
from twisted.internet import defer, error
from twisted.internet.testing import StringTransport
from twisted.trial import unittest
@defer.inlineCallbacks
def test_ControlL(self):
    """
        CTRL+L is generally used as a redraw-screen command in terminal
        applications.  Manhole doesn't currently respect this usage of it,
        but it should at least do something reasonable in response to this
        event (rather than, say, eating your face).
        """
    self._testwrite(b'\n1 + 1')
    yield self.recvlineClient.expect(b'\\+ 1')
    self._assertBuffer([b'>>> ', b'>>> 1 + 1'])
    self._testwrite(manhole.CTRL_L + b' + 1')
    yield self.recvlineClient.expect(b'1 \\+ 1 \\+ 1')
    self._assertBuffer([b'>>> 1 + 1 + 1'])