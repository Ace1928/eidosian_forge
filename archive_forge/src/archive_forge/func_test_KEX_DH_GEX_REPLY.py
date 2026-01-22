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
def test_KEX_DH_GEX_REPLY(self):
    """
        Test that the KEX_DH_GEX_REPLY message results in a verified
        server.
        """
    exchangeHash, signature, packetStart = self.begin_KEX_DH_GEX_REPLY()

    def _cbTestKEX_DH_GEX_REPLY(value):
        self.assertIsNone(value)
        self.assertTrue(self.calledVerifyHostKey)
        self.assertEqual(self.proto.sessionID, exchangeHash)
    d = self.proto.ssh_KEX_DH_GEX_REPLY(packetStart + common.NS(signature))
    d.addCallback(_cbTestKEX_DH_GEX_REPLY)
    return d