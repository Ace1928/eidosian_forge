import unittest
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import *
from Cryptodome.Math.Numbers import Integer
from Cryptodome.Math.Primality import (
def test_is_prime(self):
    primes = (170141183460469231731687303715884105727, 19175002942688032928599, 1363005552434666078217421284621279933627102780881053358473, 2 ** 521 - 1)
    for p in primes:
        self.assertEqual(test_probable_prime(p), PROBABLY_PRIME)
    not_primes = (4754868377601046732119933839981363081972014948522510826417784001, 1334733877147062382486934807105197899496002201113849920496510541601, 260849323075371835669784094383812120359260783810157225730623388382401)
    for np in not_primes:
        self.assertEqual(test_probable_prime(np), COMPOSITE)
    from Cryptodome.Util.number import sieve_base
    for p in sieve_base[:100]:
        res = test_probable_prime(p)
        self.assertEqual(res, PROBABLY_PRIME)