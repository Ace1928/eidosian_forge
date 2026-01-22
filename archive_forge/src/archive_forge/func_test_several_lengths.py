import unittest
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.number import long_to_bytes, bytes_to_long
from Cryptodome.Util.py3compat import *
from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib,
from Cryptodome.Hash import SHAKE128
from Cryptodome.Math.Numbers import Integer
from Cryptodome.Math._IntegerCustom import _raw_montgomery
from Cryptodome.Random.random import StrongRandom
def test_several_lengths(self):
    prng = SHAKE128.new().update(b('Test'))
    for length in range(1, 100):
        modulus2 = Integer.from_bytes(prng.read(length)) | 1
        base = Integer.from_bytes(prng.read(length)) % modulus2
        exponent2 = Integer.from_bytes(prng.read(length))
        expected = pow(base, exponent2, modulus2)
        result = monty_pow(base, exponent2, modulus2)
        self.assertEqual(result, expected)