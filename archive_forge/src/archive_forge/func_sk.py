import unittest
from binascii import unhexlify
from Cryptodome.PublicKey import ECC
from Cryptodome.Signature import eddsa
from Cryptodome.Hash import SHA512, SHAKE256
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors_wycheproof
from Cryptodome.Util.number import bytes_to_long
def sk(group):
    elem = group['key']['sk']
    return unhexlify(elem)