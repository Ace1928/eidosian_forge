from __future__ import annotations
import binascii
import hmac
import struct
import types
import zlib
from hashlib import md5, sha1, sha256, sha384, sha512
from typing import Any, Callable, Dict, Tuple, Union
from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import dh, ec, x25519
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from typing_extensions import Literal
from twisted import __version__ as twisted_version
from twisted.conch.ssh import _kex, address, keys
from twisted.conch.ssh.common import MP, NS, ffs, getMP, getNS
from twisted.internet import defer, protocol
from twisted.logger import Logger
from twisted.python import randbytes
from twisted.python.compat import iterbytes, networkString
def setKeys(self, outIV, outKey, inIV, inKey, outInteg, inInteg):
    """
        Set up the ciphers and hashes using the given keys,

        @param outIV: the outgoing initialization vector
        @param outKey: the outgoing encryption key
        @param inIV: the incoming initialization vector
        @param inKey: the incoming encryption key
        @param outInteg: the outgoing integrity key
        @param inInteg: the incoming integrity key.
        """
    o = self._getCipher(self.outCipType, outIV, outKey)
    self.encryptor = o.encryptor()
    self.encBlockSize = o.algorithm.block_size // 8
    o = self._getCipher(self.inCipType, inIV, inKey)
    self.decryptor = o.decryptor()
    self.decBlockSize = o.algorithm.block_size // 8
    self.outMAC = self._getMAC(self.outMACType, outInteg)
    self.inMAC = self._getMAC(self.inMACType, inInteg)
    if self.inMAC:
        self.verifyDigestSize = self.inMAC[3]