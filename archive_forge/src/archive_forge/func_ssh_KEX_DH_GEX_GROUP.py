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
def ssh_KEX_DH_GEX_GROUP(self, packet):
    """
        This handles different messages which share an integer value.

        If the key exchange does not have a fixed prime/generator group,
        we generate a Diffie-Hellman public key and send it in a
        MSG_KEX_DH_GEX_INIT message.

        Payload::
            string g (group generator)
            string p (group prime)

        @type packet: L{bytes}
        @param packet: The message data.
        """
    if _kex.isFixedGroup(self.kexAlg):
        return self._ssh_KEXDH_REPLY(packet)
    elif _kex.isEllipticCurve(self.kexAlg):
        return self._ssh_KEX_ECDH_REPLY(packet)
    else:
        self.p, rest = getMP(packet)
        self.g, rest = getMP(rest)
        self._startEphemeralDH()
        self.sendPacket(MSG_KEX_DH_GEX_INIT, self.dhSecretKeyPublicMP)