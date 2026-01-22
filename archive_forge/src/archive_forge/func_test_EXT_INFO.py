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
def test_EXT_INFO(self):
    """
        When an EXT_INFO message is received, the transport stores a mapping
        of the peer's advertised extensions.  See RFC 8308, section 2.3.
        """
    self.proto.dispatchMessage(transport.MSG_EXT_INFO, b'\x00\x00\x00\x02' + common.NS(b'server-sig-algs') + common.NS(b'ssh-rsa,rsa-sha2-256,rsa-sha2-512') + common.NS(b'no-flow-control') + common.NS(b's'))
    self.assertEqual(self.proto.peerExtensions, {b'server-sig-algs': b'ssh-rsa,rsa-sha2-256,rsa-sha2-512', b'no-flow-control': b's'})