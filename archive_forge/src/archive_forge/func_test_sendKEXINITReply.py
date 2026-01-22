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
def test_sendKEXINITReply(self):
    """
        When a KEXINIT message is received which is not a reply to an earlier
        KEXINIT message which was sent, a KEXINIT reply is sent.
        """
    self.finishKeyExchange(self.proto)
    del self.packets[:]
    self.proto.dispatchMessage(transport.MSG_KEXINIT, self._A_KEXINIT_MESSAGE)
    self.assertEqual(len(self.packets), 1)
    self.assertEqual(self.packets[0][0], transport.MSG_KEXINIT)