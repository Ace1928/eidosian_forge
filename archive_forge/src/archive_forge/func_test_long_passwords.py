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
def test_long_passwords(self):
    tvs = [(b'g*3Q45="8NNgpT&mbMJ$Omfr.#ZeW?FP=CE$#roHd?97uL0F-]`?u73c"\\[."*)qU34@VG', 4, b'T2XJ5MOWvHQZRijl8LIKkO', b'$2a$04$T2XJ5MOWvHQZRijl8LIKkOQKIyX75KBfuLsuRYOJz5OjwBNF2lM8a'), (b'\\M+*8;&QE=Ll[>5?Ui"^ai#iQH7ZFtNMfs3AROnIncE9"BNNoEgO[[*Yk8;RQ(#S,;I+aT', 5, b'wgkOlGNXIVE2fWkT3gyRoO', b'$2a$05$wgkOlGNXIVE2fWkT3gyRoOqWi4gbi1Wv2Q2Jx3xVs3apl1w.Wtj8C'), (b"M.E1=dt<.L0Q&p;94NfGm_Oo23+Kpl@M5?WIAL.[@/:'S)W96G8N^AWb7_smmC]>7#fGoB", 6, b'W9zTCl35nEvUukhhFzkKMe', b'$2a$06$W9zTCl35nEvUukhhFzkKMekjT9/pj7M0lihRVEZrX3m8/SBNZRX7i')]
    for idx, (password, cost, salt64, result) in enumerate(tvs):
        x = bcrypt(password, cost, salt=_bcrypt_decode(salt64))
        self.assertEqual(x, result)
        bcrypt_check(password, result)