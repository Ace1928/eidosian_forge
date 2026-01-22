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
def test_KEX_DH_GEX_GROUP(self):
    """
        Test that the KEX_DH_GEX_GROUP message results in a
        KEX_DH_GEX_INIT message with the client's Diffie-Hellman public key.
        """
    self.test_KEXINIT_groupexchange()
    self.proto.ssh_KEX_DH_GEX_GROUP(common.MP(self.P1536) + common.MP(2))
    self.assertEqual(self.proto.p, self.P1536)
    self.assertEqual(self.proto.g, 2)
    x = self.proto.dhSecretKey.private_numbers().x
    self.assertEqual(common.MP(x)[5:], b'\x99' * 192)
    self.assertEqual(self.proto.dhSecretKeyPublicMP, common.MP(pow(2, x, self.P1536)))
    self.assertEqual(self.packets[1:], [(transport.MSG_KEX_DH_GEX_INIT, self.proto.dhSecretKeyPublicMP)])