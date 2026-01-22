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
def test_sendVersion(self):
    """
        Test that the first thing sent over the connection is the version
        string.  The 'softwareversion' part must consist of printable
        US-ASCII characters, with the exception of whitespace characters and
        the minus sign.

        RFC 4253, section 4.2.
        """
    version = self.transport.value().split(b'\r\n', 1)[0]
    self.assertEqual(version, b'SSH-2.0-Twisted_' + twisted_version.encode('ascii'))
    softwareVersion = version.decode('ascii')[len('SSH-2.0-'):]
    softwareVersionRegex = '^(' + '|'.join((re.escape(c) for c in string.printable if c != '-' and (not c.isspace()))) + ')*$'
    self.assertRegex(softwareVersion, softwareVersionRegex)