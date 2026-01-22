import unittest
from Cryptodome.SelfTest.st_common import list_test_cases, a2b_hex, b2a_hex
from Cryptodome import Random
from Cryptodome.PublicKey import ElGamal
from Cryptodome.Util.number import bytes_to_long
from Cryptodome.Util.py3compat import *
def convert_tv(self, tv, as_longs=0):
    """Convert a test vector from textual form (hexadecimal ascii
        to either integers or byte strings."""
    key_comps = ('p', 'g', 'y', 'x')
    tv2 = {}
    for c in tv.keys():
        tv2[c] = a2b_hex(tv[c])
        if as_longs or c in key_comps or c in ('sig1', 'sig2'):
            tv2[c] = bytes_to_long(tv2[c])
    tv2['key'] = []
    for c in key_comps:
        tv2['key'] += [tv2[c]]
        del tv2[c]
    return tv2