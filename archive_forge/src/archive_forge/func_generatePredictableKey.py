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
def generatePredictableKey(transport):
    p = transport.p
    g = transport.g
    bits = p.bit_length()
    x = sum((9 << x for x in range(0, bits - 3, 4)))
    y = pow(g, x, p)
    try:
        transport.dhSecretKey = dh.DHPrivateNumbers(x, dh.DHPublicNumbers(y, dh.DHParameterNumbers(p, g))).private_key(default_backend())
    except ValueError:
        print(f'\np={p}\ng={g}\nx={x}\n')
        raise
    transport.dhSecretKeyPublicMP = common.MP(transport.dhSecretKey.public_key().public_numbers().y)