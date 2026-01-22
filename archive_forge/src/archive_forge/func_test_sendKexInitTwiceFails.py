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
def test_sendKexInitTwiceFails(self):
    """
        A new key exchange cannot be started while a key exchange is already in
        progress.  If an attempt is made to send a I{KEXINIT} message using
        L{SSHTransportBase.sendKexInit} while a key exchange is in progress
        causes that method to raise a L{RuntimeError}.
        """
    self.assertRaises(RuntimeError, self.proto.sendKexInit)