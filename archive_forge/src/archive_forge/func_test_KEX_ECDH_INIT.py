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
def test_KEX_ECDH_INIT(self):
    """
        Test that the KEXDH_INIT message causes the server to send a
        KEXDH_REPLY with the server's public key and a signature.
        """
    self.proto.supportedKeyExchanges = [self.kexAlgorithm]
    self.proto.supportedPublicKeys = [b'ssh-rsa']
    self.proto.dataReceived(self.transport.value())
    pubHostKey, privHostKey = self.proto._getHostKeys(b'ssh-rsa')
    ecPriv = self.proto._generateECPrivateKey()
    ecPub = ecPriv.public_key()
    encPub = self.proto._encodeECPublicKey(ecPub)
    self.proto.ssh_KEX_DH_GEX_REQUEST_OLD(common.NS(encPub))
    sharedSecret = self.proto._generateECSharedSecret(ecPriv, self.proto._encodeECPublicKey(self.proto.ecPub))
    h = self.hashProcessor()
    h.update(common.NS(self.proto.otherVersionString))
    h.update(common.NS(self.proto.ourVersionString))
    h.update(common.NS(self.proto.otherKexInitPayload))
    h.update(common.NS(self.proto.ourKexInitPayload))
    h.update(common.NS(pubHostKey.blob()))
    h.update(common.NS(encPub))
    h.update(common.NS(self.proto._encodeECPublicKey(self.proto.ecPub)))
    h.update(sharedSecret)
    exchangeHash = h.digest()
    signature = privHostKey.sign(exchangeHash)
    self.assertEqual(self.packets, [(transport.MSG_KEXDH_REPLY, common.NS(pubHostKey.blob()) + common.NS(self.proto._encodeECPublicKey(self.proto.ecPub)) + common.NS(signature)), (transport.MSG_NEWKEYS, b'')])