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
def simulateKeyExchange(self, sharedSecret, exchangeHash):
    """
        Finish a key exchange by calling C{_keySetup} with the given arguments.
        Also do extra whitebox stuff to satisfy that method's assumption that
        some kind of key exchange has actually taken place.
        """
    self.proto._keyExchangeState = self.proto._KEY_EXCHANGE_REQUESTED
    self.proto._blockedByKeyExchange = []
    self.proto._keySetup(sharedSecret, exchangeHash)