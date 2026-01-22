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
def test_KEX_DH_GEX_REQUEST(self, keyAlgorithm=b'ssh-rsa'):
    """
        Test that the KEX_DH_GEX_REQUEST message causes the server to reply
        with a KEX_DH_GEX_GROUP message with the correct Diffie-Hellman
        group.
        """
    self.proto.supportedKeyExchanges = [self.kexAlgorithm]
    self.proto.supportedPublicKeys = [keyAlgorithm]
    self.proto.dataReceived(self.transport.value())
    self.proto.ssh_KEX_DH_GEX_REQUEST(b'\x00\x00\x04\x00\x00\x00\x08\x00' + b'\x00\x00\x0c\x00')
    dhGenerator, dhPrime = self.proto.factory.getPrimes().get(2048)[0]
    self.assertEqual(self.packets, [(transport.MSG_KEX_DH_GEX_GROUP, common.MP(dhPrime) + b'\x00\x00\x00\x01\x02')])
    self.assertEqual(self.proto.g, 2)
    self.assertEqual(self.proto.p, dhPrime)