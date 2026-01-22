import os
import errno
import warnings
import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import bord, tostr, FileNotFoundError
from Cryptodome.Util.asn1 import DerSequence, DerBitString
from Cryptodome.Util.number import bytes_to_long
from Cryptodome.Hash import SHAKE128
from Cryptodome.PublicKey import ECC
def create_ref_keys_ed448():
    key_lines = load_file('ecc_ed448.txt').splitlines()
    seed = compact(key_lines[6:10])
    key = ECC.construct(curve='Ed448', seed=seed)
    return (key, key.public_key())