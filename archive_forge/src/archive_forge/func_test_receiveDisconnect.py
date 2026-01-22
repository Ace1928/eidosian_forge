import binascii
import re
import string
import struct
import types
from hashlib import md5, sha1, sha256, sha384, sha512
from typing import Dict, List, Optional, Tuple, Type
from twisted import __version__ as twisted_version
from twisted.conch.error import ConchError
from twisted.conch.ssh import _kex, address, service
from twisted.internet import defer
from twisted.protocols import loopback
from twisted.python import randbytes
from twisted.python.compat import iterbytes
from twisted.python.randbytes import insecureRandom
from twisted.python.reflect import requireModule
from twisted.test import proto_helpers
from twisted.trial.unittest import TestCase
def test_receiveDisconnect(self):
    """
        Test that disconnection messages are received correctly.  See
        test_sendDisconnect.
        """
    disconnected = [False]

    def stubLoseConnection():
        disconnected[0] = True
    self.transport.loseConnection = stubLoseConnection
    self.proto.dispatchMessage(transport.MSG_DISCONNECT, b'\x00\x00\x00\xff\x00\x00\x00\x04test')
    self.assertEqual(self.proto.errors, [(255, b'test')])
    self.assertTrue(disconnected[0])