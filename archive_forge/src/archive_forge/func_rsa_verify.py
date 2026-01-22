from __future__ import print_function
import base64
import hashlib
import os
from cStringIO import StringIO
from M2Crypto import BIO, EVP, RSA, X509, m2
def rsa_verify(xml, signature, key, c14n_exc=True):
    """Verify a XML document signature usign RSA-SHA1, return True if valid"""
    if key.startswith('-----BEGIN PUBLIC KEY-----'):
        bio = BIO.MemoryBuffer(key)
        rsa = RSA.load_pub_key_bio(bio)
    else:
        rsa = RSA.load_pub_key(certificate)
    pubkey = EVP.PKey()
    pubkey.assign_rsa(rsa)
    pubkey.reset_context(md='sha1')
    pubkey.verify_init()
    pubkey.verify_update(canonicalize(xml, c14n_exc))
    ret = pubkey.verify_final(base64.b64decode(signature))
    return ret == 1