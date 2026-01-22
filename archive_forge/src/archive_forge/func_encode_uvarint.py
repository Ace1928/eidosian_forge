import base64
import json
from ._versions import (VERSION_1, VERSION_2, VERSION_3)
from ._third_party import legacy_namespace, ThirdPartyCaveatInfo
from ._keys import PublicKey
from ._error import VerificationError
import macaroonbakery.checkers as checkers
import nacl.public
import six
def encode_uvarint(n, data):
    """encodes integer into variable-length format into data."""
    if n < 0:
        raise ValueError('only support positive integer')
    while True:
        this_byte = n & 127
        n >>= 7
        if n == 0:
            data.append(this_byte)
            break
        data.append(this_byte | 128)