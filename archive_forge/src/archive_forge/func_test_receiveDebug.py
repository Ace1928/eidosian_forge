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
def test_receiveDebug(self):
    """
        Test that debug messages are received correctly.  See test_sendDebug.
        """
    self.proto.dispatchMessage(transport.MSG_DEBUG, b'\x01\x00\x00\x00\x04test\x00\x00\x00\x02en')
    self.proto.dispatchMessage(transport.MSG_DEBUG, b'\x00\x00\x00\x00\x06silent\x00\x00\x00\x02en')
    self.assertEqual(self.proto.debugs, [(True, b'test', b'en'), (False, b'silent', b'en')])