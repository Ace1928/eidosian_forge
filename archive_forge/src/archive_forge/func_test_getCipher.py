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
def test_getCipher(self):
    """
        Test that the _getCipher method returns the correct cipher.
        """
    ciphers = transport.SSHCiphers(b'A', b'B', b'C', b'D')
    iv = key = b'\x00' * 16
    for cipName, (algClass, keySize, counter) in ciphers.cipherMap.items():
        cip = ciphers._getCipher(cipName, iv, key)
        if cipName == b'none':
            self.assertIsInstance(cip, transport._DummyCipher)
        else:
            self.assertIsInstance(cip.algorithm, algClass)