import re
import unittest
from binascii import hexlify, unhexlify
from Cryptodome.Util.py3compat import tobytes, bord, bchr
from Cryptodome.Hash import (SHA1, SHA224, SHA256, SHA384, SHA512,
from Cryptodome.Signature import DSS
from Cryptodome.PublicKey import DSA, ECC
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors, load_test_vectors_wycheproof
from Cryptodome.Util.number import bytes_to_long, long_to_bytes
def test_data_rfc6979_p192(self):
    signer = DSS.new(self.key_priv_p192, 'deterministic-rfc6979')
    for message, k, r, s, module in self.signatures_p192:
        hash_obj = module.new(message)
        result = signer.sign(hash_obj)
        self.assertEqual(r + s, result)