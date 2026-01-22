import unittest
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.number import long_to_bytes, bytes_to_long
from Cryptodome.Util.py3compat import *
from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib,
from Cryptodome.Hash import SHAKE128
from Cryptodome.Math.Numbers import Integer
from Cryptodome.Math._IntegerCustom import _raw_montgomery
from Cryptodome.Random.random import StrongRandom
def test_stress_65(self):
    prng = create_rng(b('Test 65'))
    length = 65
    for _ in range(2000):
        modulus = prng.getrandbits(8 * length) | 1
        base = prng.getrandbits(8 * length) % modulus
        exponent = prng.getrandbits(8 * length)
        expected = pow(base, exponent, modulus)
        result = monty_pow(base, exponent, modulus)
        self.assertEqual(result, expected)