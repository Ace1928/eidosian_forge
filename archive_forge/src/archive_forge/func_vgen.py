import operator
import struct
from passlib.utils.compat import izip
from passlib.crypto.digest import pbkdf2_hmac
from passlib.crypto.scrypt._salsa import salsa20
def vgen():
    i = 0
    while i < n:
        last = tuple(buffer)
        yield last
        bmix(last, buffer)
        i += 1