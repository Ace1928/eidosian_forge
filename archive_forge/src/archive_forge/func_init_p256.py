from __future__ import print_function
import re
import struct
import binascii
from collections import namedtuple
from Cryptodome.Util.py3compat import bord, tobytes, tostr, bchr, is_string
from Cryptodome.Util.number import bytes_to_long, long_to_bytes
from Cryptodome.Math.Numbers import Integer
from Cryptodome.Util.asn1 import (DerObjectId, DerOctetString, DerSequence,
from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib, VoidPointer,
from Cryptodome.PublicKey import (_expand_subject_public_key_info,
from Cryptodome.Hash import SHA512, SHAKE256
from Cryptodome.Random import get_random_bytes
from Cryptodome.Random.random import getrandbits
def init_p256():
    p = 115792089210356248762697446949407573530086143415290314195533631308867097853951
    b = 41058363725152142129326129780047268409114441015993725554835256314039467401291
    order = 115792089210356248762697446949407573529996955224135760342422259061068512044369
    Gx = 48439561293906451759052585252797914202762949526041747995844080717082404635286
    Gy = 36134250956749795798585127919587881956611106672985015071877198253568414405109
    p256_modulus = long_to_bytes(p, 32)
    p256_b = long_to_bytes(b, 32)
    p256_order = long_to_bytes(order, 32)
    ec_p256_context = VoidPointer()
    result = _ec_lib.ec_ws_new_context(ec_p256_context.address_of(), c_uint8_ptr(p256_modulus), c_uint8_ptr(p256_b), c_uint8_ptr(p256_order), c_size_t(len(p256_modulus)), c_ulonglong(getrandbits(64)))
    if result:
        raise ImportError('Error %d initializing P-256 context' % result)
    context = SmartPointer(ec_p256_context.get(), _ec_lib.ec_free_context)
    p256 = _Curve(Integer(p), Integer(b), Integer(order), Integer(Gx), Integer(Gy), None, 256, '1.2.840.10045.3.1.7', context, 'NIST P-256', 'ecdsa-sha2-nistp256', 'p256')
    global p256_names
    _curves.update(dict.fromkeys(p256_names, p256))