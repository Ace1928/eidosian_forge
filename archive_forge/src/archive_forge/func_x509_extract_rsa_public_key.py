from __future__ import print_function
import base64
import hashlib
import os
from cStringIO import StringIO
from M2Crypto import BIO, EVP, RSA, X509, m2
def x509_extract_rsa_public_key(cert, binary=False):
    """Return the public key (PEM format) from a X509 certificate"""
    x509 = x509_parse_cert(cert, binary)
    return x509.get_pubkey().get_rsa().as_pem()