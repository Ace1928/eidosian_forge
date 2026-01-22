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
def init_p384():
    p = 39402006196394479212279040100143613805079739270465446667948293404245721771496870329047266088258938001861606973112319
    b = 27580193559959705877849011840389048093056905856361568521428707301988689241309860865136260764883745107765439761230575
    order = 39402006196394479212279040100143613805079739270465446667946905279627659399113263569398956308152294913554433653942643
    Gx = 26247035095799689268623156744566981891852923491109213387815615900925518854738050089022388053975719786650872476732087
    Gy = 8325710961489029985546751289520108179287853048861315594709205902480503199884419224438643760392947333078086511627871
    p384_modulus = long_to_bytes(p, 48)
    p384_b = long_to_bytes(b, 48)
    p384_order = long_to_bytes(order, 48)
    ec_p384_context = VoidPointer()
    result = _ec_lib.ec_ws_new_context(ec_p384_context.address_of(), c_uint8_ptr(p384_modulus), c_uint8_ptr(p384_b), c_uint8_ptr(p384_order), c_size_t(len(p384_modulus)), c_ulonglong(getrandbits(64)))
    if result:
        raise ImportError('Error %d initializing P-384 context' % result)
    context = SmartPointer(ec_p384_context.get(), _ec_lib.ec_free_context)
    p384 = _Curve(Integer(p), Integer(b), Integer(order), Integer(Gx), Integer(Gy), None, 384, '1.3.132.0.34', context, 'NIST P-384', 'ecdsa-sha2-nistp384', 'p384')
    global p384_names
    _curves.update(dict.fromkeys(p384_names, p384))