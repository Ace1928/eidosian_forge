import unittest
from Cryptodome.SelfTest.st_common import list_test_cases, a2b_hex
from Cryptodome.SelfTest.loader import load_test_vectors_wycheproof
from Cryptodome.PublicKey import RSA
from Cryptodome.Cipher import PKCS1_OAEP as PKCS
from Cryptodome.Hash import MD2, MD5, SHA1, SHA256, RIPEMD160, SHA224, SHA384, SHA512
from Cryptodome import Random
from Cryptodome.Signature.pss import MGF1
from Cryptodome.Util.py3compat import b, bchr
def testEncryptDecrypt4(self):
    global mgfcalls

    def newMGF(seed, maskLen):
        global mgfcalls
        mgfcalls += 1
        return b'\x00' * maskLen
    mgfcalls = 0
    pt = self.rng(32)
    cipher = PKCS.new(self.key1024, mgfunc=newMGF)
    ct = cipher.encrypt(pt)
    self.assertEqual(mgfcalls, 2)
    self.assertEqual(cipher.decrypt(ct), pt)