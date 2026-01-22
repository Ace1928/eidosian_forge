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
def test_same_password_and_random_salt(self):
    tvs = [(b'Q/A:k3DP;X@=<0"hg&9c', 4, b'wbgDTvLMtyjQlNK7fjqwyO', b'$2a$04$wbgDTvLMtyjQlNK7fjqwyOakBoACQuYh11.VsKNarF4xUIOBWgD6S'), (b'Q/A:k3DP;X@=<0"hg&9c', 5, b'zbAaOmloOhxiKItjznRqru', b'$2a$05$zbAaOmloOhxiKItjznRqrunRqHlu3MAa7pMGv26Rr3WwyfGcwoRm6'), (b'Q/A:k3DP;X@=<0"hg&9c', 6, b'aOK0bWUvLI0qLkc3ti5jyu', b'$2a$06$aOK0bWUvLI0qLkc3ti5jyuAIQoqRzuqoK09kQqQ6Ou/YKDhW50/qa')]
    for idx, (password, cost, salt64, result) in enumerate(tvs):
        x = bcrypt(password, cost, salt=_bcrypt_decode(salt64))
        self.assertEqual(x, result)
        bcrypt_check(password, result)