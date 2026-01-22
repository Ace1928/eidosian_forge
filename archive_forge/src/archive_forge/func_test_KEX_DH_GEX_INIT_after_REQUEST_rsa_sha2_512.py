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
def test_KEX_DH_GEX_INIT_after_REQUEST_rsa_sha2_512(self):
    """
        Test that the KEX_DH_GEX_INIT message after the client sends
        KEX_DH_GEX_REQUEST using a public key signature algorithm other than
        the default for the public key format causes the server to send a
        KEX_DH_GEX_INIT message with a public key and signature.
        """
    self.test_KEX_DH_GEX_REQUEST(keyAlgorithm=b'rsa-sha2-512')
    pubHostKey, privHostKey = self.proto._getHostKeys(b'rsa-sha2-512')
    e = pow(self.proto.g, 3, self.proto.p)
    y = common.getMP(b'\x00\x00\x01\x00' + b'\x99' * 256)[0]
    f = _MPpow(self.proto.g, y, self.proto.p)
    sharedSecret = _MPpow(e, y, self.proto.p)
    h = self.hashProcessor()
    h.update(common.NS(self.proto.ourVersionString) * 2)
    h.update(common.NS(self.proto.ourKexInitPayload) * 2)
    h.update(common.NS(pubHostKey.blob()))
    h.update(b'\x00\x00\x04\x00\x00\x00\x08\x00\x00\x00\x0c\x00')
    h.update(common.MP(self.proto.p))
    h.update(common.MP(self.proto.g))
    h.update(common.MP(e))
    h.update(f)
    h.update(sharedSecret)
    exchangeHash = h.digest()
    self.proto.ssh_KEX_DH_GEX_INIT(common.MP(e))
    self.assertEqual(self.packets[1], (transport.MSG_KEX_DH_GEX_REPLY, common.NS(pubHostKey.blob()) + f + common.NS(privHostKey.sign(exchangeHash, signatureType=b'rsa-sha2-512'))))