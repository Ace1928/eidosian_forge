import unittest
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.number import long_to_bytes, bytes_to_long
from Cryptodome.Util.py3compat import *
from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib,
from Cryptodome.Hash import SHAKE128
from Cryptodome.Math.Numbers import Integer
from Cryptodome.Math._IntegerCustom import _raw_montgomery
from Cryptodome.Random.random import StrongRandom
def test_zero_modulus(self):
    base = 100433627766186892221372630771322662657637687111424552206335
    self.assertRaises(ExceptionModulus, monty_pow, base, exponent1, 0)
    self.assertRaises(ExceptionModulus, monty_pow, 0, 0, 0)