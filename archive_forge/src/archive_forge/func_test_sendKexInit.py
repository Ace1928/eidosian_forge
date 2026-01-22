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
def test_sendKexInit(self):
    """
        Test that the KEXINIT (key exchange initiation) message is sent
        correctly.  Payload::
            bytes[16] cookie
            string key exchange algorithms
            string public key algorithms
            string outgoing ciphers
            string incoming ciphers
            string outgoing MACs
            string incoming MACs
            string outgoing compressions
            string incoming compressions
            bool first packet follows
            uint32 0
        """
    value = self.transport.value().split(b'\r\n', 1)[1]
    self.proto.buf = value
    packet = self.proto.getPacket()
    self.assertEqual(packet[0:1], bytes((transport.MSG_KEXINIT,)))
    self.assertEqual(packet[1:17], b'\x99' * 16)
    keyExchanges, pubkeys, ciphers1, ciphers2, macs1, macs2, compressions1, compressions2, languages1, languages2, buf = common.getNS(packet[17:], 10)
    self.assertEqual(keyExchanges, b','.join(self.proto.supportedKeyExchanges + [b'ext-info-s']))
    self.assertEqual(pubkeys, b','.join(self.proto.supportedPublicKeys))
    self.assertEqual(ciphers1, b','.join(self.proto.supportedCiphers))
    self.assertEqual(ciphers2, b','.join(self.proto.supportedCiphers))
    self.assertEqual(macs1, b','.join(self.proto.supportedMACs))
    self.assertEqual(macs2, b','.join(self.proto.supportedMACs))
    self.assertEqual(compressions1, b','.join(self.proto.supportedCompressions))
    self.assertEqual(compressions2, b','.join(self.proto.supportedCompressions))
    self.assertEqual(languages1, b','.join(self.proto.supportedLanguages))
    self.assertEqual(languages2, b','.join(self.proto.supportedLanguages))
    self.assertEqual(buf, b'\x00' * 5)