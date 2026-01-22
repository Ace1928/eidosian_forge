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
def test_buildProtocolNoPrimes(self):
    """
        Group key exchanges are not supported when we don't have the primes
        database.
        """
    f1 = self.makeSSHFactory(primes=None)
    p1 = f1.buildProtocol(None)
    self.assertNotIn(b'diffie-hellman-group-exchange-sha1', p1.supportedKeyExchanges)
    self.assertNotIn(b'diffie-hellman-group-exchange-sha256', p1.supportedKeyExchanges)