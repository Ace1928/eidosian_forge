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
def isVerified(self, direction='out'):
    """
        Check if the connection is verified/authentication in the given direction.

        @type direction: L{str}
        @param direction: The direction: one of 'out', 'in', or 'both'.

        @rtype: L{bool}
        @return: C{True} if it is verified.
        """
    if direction == 'out':
        return self.currentEncryptions.outMACType != b'none'
    elif direction == 'in':
        return self.currentEncryptions.inMACType != b'none'
    elif direction == 'both':
        return self.isVerified('in') and self.isVerified('out')
    else:
        raise TypeError('direction must be "out", "in", or "both"')