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
def init_p192():
    p = 6277101735386680763835789423207666416083908700390324961279
    b = 2455155546008943817740293915197451784769108058161191238065
    order = 6277101735386680763835789423176059013767194773182842284081
    Gx = 602046282375688656758213480587526111916698976636884684818
    Gy = 174050332293622031404857552280219410364023488927386650641
    p192_modulus = long_to_bytes(p, 24)
    p192_b = long_to_bytes(b, 24)
    p192_order = long_to_bytes(order, 24)
    ec_p192_context = VoidPointer()
    result = _ec_lib.ec_ws_new_context(ec_p192_context.address_of(), c_uint8_ptr(p192_modulus), c_uint8_ptr(p192_b), c_uint8_ptr(p192_order), c_size_t(len(p192_modulus)), c_ulonglong(getrandbits(64)))
    if result:
        raise ImportError('Error %d initializing P-192 context' % result)
    context = SmartPointer(ec_p192_context.get(), _ec_lib.ec_free_context)
    p192 = _Curve(Integer(p), Integer(b), Integer(order), Integer(Gx), Integer(Gy), None, 192, '1.2.840.10045.3.1.1', context, 'NIST P-192', 'ecdsa-sha2-nistp192', 'p192')
    global p192_names
    _curves.update(dict.fromkeys(p192_names, p192))