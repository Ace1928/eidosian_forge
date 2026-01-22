import re
import unittest
from binascii import unhexlify
from Cryptodome.Util.py3compat import b, bchr
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors, load_test_vectors_wycheproof
from Cryptodome.Hash import SHA1, HMAC, SHA256, MD5, SHA224, SHA384, SHA512
from Cryptodome.Cipher import AES, DES3
from Cryptodome.Protocol.KDF import (PBKDF1, PBKDF2, _S2V, HKDF, scrypt,
from Cryptodome.Protocol.KDF import _bcrypt_decode
def test_increasing_password_length(self):
    tvs = [(b'a', 4, b'5DCebwootqWMCp59ISrMJ.', b'$2a$04$5DCebwootqWMCp59ISrMJ.l4WvgHIVg17ZawDIrDM2IjlE64GDNQS'), (b'aa', 4, b'5DCebwootqWMCp59ISrMJ.', b'$2a$04$5DCebwootqWMCp59ISrMJ.AyUxBk.ThHlsLvRTH7IqcG7yVHJ3SXq'), (b'aaa', 4, b'5DCebwootqWMCp59ISrMJ.', b'$2a$04$5DCebwootqWMCp59ISrMJ.BxOVac5xPB6XFdRc/ZrzM9FgZkqmvbW'), (b'aaaa', 4, b'5DCebwootqWMCp59ISrMJ.', b'$2a$04$5DCebwootqWMCp59ISrMJ.Qbr209bpCtfl5hN7UQlG/L4xiD3AKau'), (b'aaaaa', 4, b'5DCebwootqWMCp59ISrMJ.', b'$2a$04$5DCebwootqWMCp59ISrMJ.oWszihPjDZI0ypReKsaDOW1jBl7oOii'), (b'aaaaaa', 4, b'5DCebwootqWMCp59ISrMJ.', b'$2a$04$5DCebwootqWMCp59ISrMJ./k.Xxn9YiqtV/sxh3EHbnOHd0Qsq27K'), (b'aaaaaaa', 4, b'5DCebwootqWMCp59ISrMJ.', b'$2a$04$5DCebwootqWMCp59ISrMJ.PYJqRFQbgRbIjMd5VNKmdKS4sBVOyDe'), (b'aaaaaaaa', 4, b'5DCebwootqWMCp59ISrMJ.', b'$2a$04$5DCebwootqWMCp59ISrMJ..VMYfzaw1wP/SGxowpLeGf13fxCCt.q'), (b'aaaaaaaaa', 4, b'5DCebwootqWMCp59ISrMJ.', b'$2a$04$5DCebwootqWMCp59ISrMJ.5B0p054nO5WgAD1n04XslDY/bqY9RJi'), (b'aaaaaaaaaa', 4, b'5DCebwootqWMCp59ISrMJ.', b'$2a$04$5DCebwootqWMCp59ISrMJ.INBTgqm7sdlBJDg.J5mLMSRK25ri04y'), (b'aaaaaaaaaaa', 4, b'5DCebwootqWMCp59ISrMJ.', b'$2a$04$5DCebwootqWMCp59ISrMJ.s3y7CdFD0OR5p6rsZw/eZ.Dla40KLfm'), (b'aaaaaaaaaaaa', 4, b'5DCebwootqWMCp59ISrMJ.', b'$2a$04$5DCebwootqWMCp59ISrMJ.Jx742Djra6Q7PqJWnTAS.85c28g.Siq'), (b'aaaaaaaaaaaaa', 4, b'5DCebwootqWMCp59ISrMJ.', b'$2a$04$5DCebwootqWMCp59ISrMJ.oKMXW3EZcPHcUV0ib5vDBnh9HojXnLu'), (b'aaaaaaaaaaaaaa', 4, b'5DCebwootqWMCp59ISrMJ.', b'$2a$04$5DCebwootqWMCp59ISrMJ.w6nIjWpDPNSH5pZUvLjC1q25ONEQpeS'), (b'aaaaaaaaaaaaaaa', 4, b'5DCebwootqWMCp59ISrMJ.', b'$2a$04$5DCebwootqWMCp59ISrMJ.k1b2/r9A/hxdwKEKurg6OCn4MwMdiGq'), (b'aaaaaaaaaaaaaaaa', 4, b'5DCebwootqWMCp59ISrMJ.', b'$2a$04$5DCebwootqWMCp59ISrMJ.3prCNHVX1Ws.7Hm2bJxFUnQOX9f7DFa')]
    for idx, (password, cost, salt64, result) in enumerate(tvs):
        x = bcrypt(password, cost, salt=_bcrypt_decode(salt64))
        self.assertEqual(x, result)
        bcrypt_check(password, result)