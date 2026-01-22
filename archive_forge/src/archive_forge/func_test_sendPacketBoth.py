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
def test_sendPacketBoth(self):
    """
        Test that packets sent while compression and encryption are
        enabled are sent correctly.  The packet type and data should be
        compressed and then the whole packet should be encrypted.
        """
    proto = MockTransportBase()
    proto.makeConnection(self.transport)
    self.finishKeyExchange(proto)
    proto.currentEncryptions = testCipher = MockCipher()
    proto.outgoingCompression = MockCompression()
    message = ord('A')
    payload = b'BC'
    self.transport.clear()
    proto.sendPacket(message, payload)
    self.assertTrue(testCipher.usedEncrypt)
    value = self.transport.value()
    self.assertEqual(value, b'\x00\x00\x00\x0e\tCBAf\x99\x99\x99\x99\x99\x99\x99\x99\x99\x02')