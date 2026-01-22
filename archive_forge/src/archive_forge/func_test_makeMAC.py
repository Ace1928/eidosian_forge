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
def test_makeMAC(self):
    """
        L{SSHCiphers.makeMAC} computes the HMAC of an outgoing SSH message with
        a particular sequence id and content data.
        """
    vectors = [(b'\x0b' * 16, b'Hi There', b'9294727a3638bb1c13f48ef8158bfc9d'), (b'Jefe', b'what do ya want for nothing?', b'750c783e6ab0b503eaa86e310a5db738'), (b'\xaa' * 16, b'\xdd' * 50, b'56be34521d144c88dbb8c733f0e8b3f6')]
    for key, data, mac in vectors:
        outMAC = transport.SSHCiphers(b'none', b'none', b'hmac-md5', b'none')
        outMAC.outMAC = outMAC._getMAC(b'hmac-md5', key)
        seqid, = struct.unpack('>L', data[:4])
        shortened = data[4:]
        self.assertEqual(mac, binascii.hexlify(outMAC.makeMAC(seqid, shortened)), f'Failed HMAC test vector; key={key!r} data={data!r}')