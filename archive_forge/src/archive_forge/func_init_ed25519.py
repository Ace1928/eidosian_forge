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
def init_ed25519():
    p = 57896044618658097711785492504343953926634992332820282019728792003956564819949
    order = 7237005577332262213973186563042994240857116359379907606001950938285454250989
    Gx = 15112221349535400772501151409588531511454012693041857206046113283949847762202
    Gy = 46316835694926478169428394003475163141307993866256225615783033603165251855960
    ed25519 = _Curve(Integer(p), None, Integer(order), Integer(Gx), Integer(Gy), None, 255, '1.3.101.112', None, 'Ed25519', 'ssh-ed25519', 'ed25519')
    global ed25519_names
    _curves.update(dict.fromkeys(ed25519_names, ed25519))