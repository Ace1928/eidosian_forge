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
def test_sendPacketPlain(self):
    """
        Test that plain (unencrypted, uncompressed) packets are sent
        correctly.  The format is::
            uint32 length (including type and padding length)
            byte padding length
            byte type
            bytes[length-padding length-2] data
            bytes[padding length] padding
        """
    proto = MockTransportBase()
    proto.makeConnection(self.transport)
    self.finishKeyExchange(proto)
    self.transport.clear()
    message = ord('A')
    payload = b'BCDEFG'
    proto.sendPacket(message, payload)
    value = self.transport.value()
    self.assertEqual(value, b'\x00\x00\x00\x0c\x04ABCDEFG\x99\x99\x99\x99')