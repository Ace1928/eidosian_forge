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
def test_random_password_and_salt_short_pw(self):
    tvs = [(b"<.S.2K(Zq'", 4, b'VYAclAMpaXY/oqAo9yUpku', b'$2a$04$VYAclAMpaXY/oqAo9yUpkuWmoYywaPzyhu56HxXpVltnBIfmO9tgu'), (b'5.rApO%5jA', 5, b'kVNDrnYKvbNr5AIcxNzeIu', b'$2a$05$kVNDrnYKvbNr5AIcxNzeIuRcyIF5cZk6UrwHGxENbxP5dVv.WQM/G'), (b'oW++kSrQW^', 6, b'QLKkRMH9Am6irtPeSKN5sO', b'$2a$06$QLKkRMH9Am6irtPeSKN5sObJGr3j47cO6Pdf5JZ0AsJXuze0IbsNm'), (b'ggJ\\KbTnDG', 7, b'4H896R09bzjhapgCPS/LYu', b'$2a$07$4H896R09bzjhapgCPS/LYuMzAQluVgR5iu/ALF8L8Aln6lzzYXwbq'), (b'49b0:;VkH/', 8, b'hfvO2retKrSrx5f2RXikWe', b'$2a$08$hfvO2retKrSrx5f2RXikWeFWdtSesPlbj08t/uXxCeZoHRWDz/xFe'), (b">9N^5jc##'", 9, b'XZLvl7rMB3EvM0c1.JHivu', b'$2a$09$XZLvl7rMB3EvM0c1.JHivuIDPJWeNJPTVrpjZIEVRYYB/mF6cYgJK'), (b'\\$ch)s4WXp', 10, b'aIjpMOLK5qiS9zjhcHR5TO', b'$2a$10$aIjpMOLK5qiS9zjhcHR5TOU7v2NFDmcsBmSFDt5EHOgp/jeTF3O/q'), (b'RYoj\\_>2P7', 12, b'esIAHiQAJNNBrsr5V13l7.', b'$2a$12$esIAHiQAJNNBrsr5V13l7.RFWWJI2BZFtQlkFyiWXjou05GyuREZa')]
    for idx, (password, cost, salt64, result) in enumerate(tvs):
        x = bcrypt(password, cost, salt=_bcrypt_decode(salt64))
        self.assertEqual(x, result)
        bcrypt_check(password, result)