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
def test_disconnectKEX_ECDH_REPLYBadSignature(self):
    """
        Test that KEX_ECDH_REPLY disconnects if the signature is bad.
        """
    kexmsg = b'\xaa' * 16 + common.NS(b'ecdh-sha2-nistp256') + common.NS(b'ssh-rsa') + common.NS(b'aes256-ctr') + common.NS(b'aes256-ctr') + common.NS(b'hmac-sha1') + common.NS(b'hmac-sha1') + common.NS(b'none') + common.NS(b'none') + common.NS(b'') + common.NS(b'') + b'\x00' + b'\x00\x00\x00\x00'
    self.proto.ssh_KEXINIT(kexmsg)
    self.proto.dataReceived(b'SSH-2.0-OpenSSH\r\n')
    self.proto.ecPriv = ec.generate_private_key(ec.SECP256R1(), default_backend())
    self.proto.ecPub = self.proto.ecPriv.public_key()
    thisPriv = ec.generate_private_key(ec.SECP256R1(), default_backend())
    thisPub = thisPriv.public_key()
    encPub = thisPub.public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
    self.proto.curve = ec.SECP256R1()
    self.proto.kexAlg = b'ecdh-sha2-nistp256'
    self.proto._ssh_KEX_ECDH_REPLY(common.NS(MockFactory().getPublicKeys()[b'ssh-rsa'].blob()) + common.NS(encPub) + common.NS(b'bad-signature'))
    self.checkDisconnected(transport.DISCONNECT_KEY_EXCHANGE_FAILED)