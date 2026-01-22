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
def test_checkBad_KEX_INIT_CurveName(self):
    """
        Test that if the server received a bad name for a curve
        we raise an UnsupportedAlgorithm error.
        """
    kexmsg = b'\xaa' * 16 + common.NS(b'ecdh-sha2-nistp256') + common.NS(b'ssh-rsa') + common.NS(b'aes256-ctr') + common.NS(b'aes256-ctr') + common.NS(b'hmac-sha1') + common.NS(b'hmac-sha1') + common.NS(b'none') + common.NS(b'none') + common.NS(b'') + common.NS(b'') + b'\x00' + b'\x00\x00\x00\x00'
    self.proto.ssh_KEXINIT(kexmsg)
    self.assertRaises(AttributeError)
    self.assertRaises(UnsupportedAlgorithm)