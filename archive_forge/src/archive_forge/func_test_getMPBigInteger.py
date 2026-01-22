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
def test_getMPBigInteger(self):
    """
        L{common.getMP} should be able to parse a big enough integer
        (that doesn't fit on one byte).
        """
    self.assertEqual(self.getMP(b'\x00\x00\x00\x04\x01\x02\x03\x04'), (16909060, b''))