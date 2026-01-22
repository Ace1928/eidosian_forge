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
def test_controlD(self):
    """
        A CTRL+D in the middle of a line doesn't close a connection,
        but at the beginning of a line it does.
        """
    self._testwrite(b'1 + 1')
    yield self.recvlineClient.expect(b'\\+ 1')
    self._assertBuffer([b'>>> 1 + 1'])
    self._testwrite(manhole.CTRL_D + b' + 1')
    yield self.recvlineClient.expect(b'\\+ 1')
    self._assertBuffer([b'>>> 1 + 1 + 1'])
    self._testwrite(b'\n')
    yield self.recvlineClient.expect(b'3\n>>> ')
    self._testwrite(manhole.CTRL_D)
    d = self.recvlineClient.onDisconnection
    yield self.assertFailure(d, error.ConnectionDone)