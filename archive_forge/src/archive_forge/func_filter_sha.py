import json
import unittest
from binascii import unhexlify
from Cryptodome.Util.py3compat import bchr
from Cryptodome.Util.number import bytes_to_long
from Cryptodome.Util.strxor import strxor
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors, load_test_vectors_wycheproof
from Cryptodome.Hash import (SHA1, SHA224, SHA256, SHA384, SHA512, SHA3_384,
from Cryptodome.PublicKey import RSA
from Cryptodome.Signature import pkcs1_15
from Cryptodome.Signature import PKCS1_v1_5
from Cryptodome.Util._file_system import pycryptodome_filename
from Cryptodome.Util.strxor import strxor
def filter_sha(group):
    hash_name = group['sha']
    if hash_name == 'SHA-512':
        return SHA512
    elif hash_name == 'SHA-512/224':
        return SHA512.new(truncate='224')
    elif hash_name == 'SHA-512/256':
        return SHA512.new(truncate='256')
    elif hash_name == 'SHA3-512':
        return SHA3_512
    elif hash_name == 'SHA-384':
        return SHA384
    elif hash_name == 'SHA3-384':
        return SHA3_384
    elif hash_name == 'SHA-256':
        return SHA256
    elif hash_name == 'SHA3-256':
        return SHA3_256
    elif hash_name == 'SHA-224':
        return SHA224
    elif hash_name == 'SHA3-224':
        return SHA3_224
    elif hash_name == 'SHA-1':
        return SHA1
    else:
        raise ValueError('Unknown hash algorithm: ' + hash_name)