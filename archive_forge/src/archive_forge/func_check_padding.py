import struct
from Cryptodome.Cipher import AES
from Cryptodome.Hash import SHA512
from Cryptodome.Protocol.KDF import _bcrypt_hash
from Cryptodome.Util.strxor import strxor
from Cryptodome.Util.py3compat import tostr, bchr, bord
def check_padding(pad):
    for v, x in enumerate(pad):
        if bord(x) != v + 1 & 255:
            raise ValueError('Incorrect padding')