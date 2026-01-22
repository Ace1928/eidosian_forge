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
def ssh_KEX_DH_GEX_REQUEST(self, packet):
    """
        Called when we receive a MSG_KEX_DH_GEX_REQUEST message.  Payload::
            integer minimum
            integer ideal
            integer maximum

        The client is asking for a Diffie-Hellman group between minimum and
        maximum size, and close to ideal if possible.  We reply with a
        MSG_KEX_DH_GEX_GROUP message.

        If we were told to ignore the next key exchange packet by ssh_KEXINIT,
        drop it on the floor and return.

        @type packet: L{bytes}
        @param packet: The message data.
        """
    if self.ignoreNextPacket:
        self.ignoreNextPacket = 0
        return
    self.dhGexRequest = packet
    min, ideal, max = struct.unpack('>3L', packet)
    self.g, self.p = self.factory.getDHPrime(ideal)
    self._startEphemeralDH()
    self.sendPacket(MSG_KEX_DH_GEX_GROUP, MP(self.p) + MP(self.g))