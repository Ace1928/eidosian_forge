from binascii import hexlify
from Cryptodome.Hash import SHA512
from .common import make_hash_tests
from Cryptodome.SelfTest.loader import load_test_vectors
def get_tests_SHA512_256():
    test_vectors = load_test_vectors(('Hash', 'SHA2'), 'SHA512_256ShortMsg.rsp', 'KAT SHA-512/256', {'len': lambda x: int(x)}) or []
    test_data = []
    for tv in test_vectors:
        try:
            if tv.startswith('['):
                continue
        except AttributeError:
            pass
        if tv.len == 0:
            tv.msg = b''
        test_data.append((hexlify(tv.md), tv.msg, tv.desc))
    tests = make_hash_tests(SHA512, 'SHA512/256', test_data, digest_size=32, oid='2.16.840.1.101.3.4.2.6', extra_params={'truncate': '256'})
    return tests