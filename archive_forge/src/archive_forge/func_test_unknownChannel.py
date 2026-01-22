import struct
from itertools import chain
from typing import Dict, List, Tuple
from twisted.conch.test.keydata import (
from twisted.conch.test.loopback import LoopbackRelay
from twisted.cred import portal
from twisted.cred.error import UnauthorizedLogin
from twisted.internet import defer, protocol, reactor
from twisted.internet.error import ProcessTerminated
from twisted.python import failure, log
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.python import components
def test_unknownChannel(self):
    """
        When an attempt is made to open an unknown channel type, the L{Deferred}
        returned by L{SSHChannel.sendRequest} fires its errback.
        """
    d = self.assertFailure(self._ourServerOurClientTest(b'crazy-unknown-channel'), Exception)

    def cbFailed(ignored):
        errors = self.flushLoggedErrors(error.ConchError)
        self.assertEqual(errors[0].value.args, (3, 'unknown channel'))
        self.assertEqual(len(errors), 1)
    d.addCallback(cbFailed)
    return d