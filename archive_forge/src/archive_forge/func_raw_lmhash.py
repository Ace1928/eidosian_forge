from warnings import warn
from binascii import hexlify
from passlib.utils.compat import unicode
from passlib.crypto.des import des_encrypt_block
from passlib.hash import nthash
def raw_lmhash(secret, encoding='ascii', hex=False):
    """encode password using des-based LMHASH algorithm; returns string of raw bytes, or unicode hex"""
    if isinstance(secret, unicode):
        secret = secret.encode(encoding)
    ns = secret.upper()[:14] + b'\x00' * (14 - len(secret))
    out = des_encrypt_block(ns[:7], LM_MAGIC) + des_encrypt_block(ns[7:], LM_MAGIC)
    return hexlify(out).decode('ascii') if hex else out