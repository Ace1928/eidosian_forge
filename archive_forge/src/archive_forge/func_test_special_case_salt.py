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
def test_special_case_salt(self):
    tvs = [('-O_=*N!2JP', 4, b'......................', b'$2a$04$......................JjuKLOX9OOwo5PceZZXSkaLDvdmgb82'), ('7B[$Q<4b>U', 5, b'......................', b'$2a$05$......................DRiedDQZRL3xq5A5FL8y7/6NM8a2Y5W'), ('>d5-I_8^.h', 6, b'......................', b'$2a$06$......................5Mq1Ng8jgDY.uHNU4h5p/x6BedzNH2W'), (')V`/UM/]1t', 4, b'.OC/.OC/.OC/.OC/.OC/.O', b'$2a$04$.OC/.OC/.OC/.OC/.OC/.OQIvKRDAam.Hm5/IaV/.hc7P8gwwIbmi'), (':@t2.bWuH]', 5, b'.OC/.OC/.OC/.OC/.OC/.O', b'$2a$05$.OC/.OC/.OC/.OC/.OC/.ONDbUvdOchUiKmQORX6BlkPofa/QxW9e'), ('b(#KljF5s"', 6, b'.OC/.OC/.OC/.OC/.OC/.O', b'$2a$06$.OC/.OC/.OC/.OC/.OC/.OHfTd9e7svOu34vi1PCvOcAEq07ST7.K'), ('@3YaJ^Xs]*', 4, b'eGA.eGA.eGA.eGA.eGA.e.', b'$2a$04$eGA.eGA.eGA.eGA.eGA.e.stcmvh.R70m.0jbfSFVxlONdj1iws0C'), ('\'"5\\!k*C(p', 5, b'eGA.eGA.eGA.eGA.eGA.e.', b'$2a$05$eGA.eGA.eGA.eGA.eGA.e.vR37mVSbfdHwu.F0sNMvgn8oruQRghy'), ("edEu7C?$'W", 6, b'eGA.eGA.eGA.eGA.eGA.e.', b'$2a$06$eGA.eGA.eGA.eGA.eGA.e.tSq0FN8MWHQXJXNFnHTPQKtA.n2a..G'), ('N7dHmg\\PI^', 4, b'999999999999999999999u', b'$2a$04$999999999999999999999uCZfA/pLrlyngNDMq89r1uUk.bQ9icOu'), ('"eJuHh!)7*', 5, b'999999999999999999999u', b'$2a$05$999999999999999999999uj8Pfx.ufrJFAoWFLjapYBS5vVEQQ/hK'), ('ZeDRJ:_tu:', 6, b'999999999999999999999u', b'$2a$06$999999999999999999999u6RB0P9UmbdbQgjoQFEJsrvrKe.BoU6q')]
    for idx, (password, cost, salt64, result) in enumerate(tvs):
        x = bcrypt(password, cost, salt=_bcrypt_decode(salt64))
        self.assertEqual(x, result)
        bcrypt_check(password, result)