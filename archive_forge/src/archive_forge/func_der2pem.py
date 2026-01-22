import os
import re
import errno
import warnings
import unittest
from Cryptodome.PublicKey import RSA
from Cryptodome.SelfTest.st_common import a2b_hex, list_test_cases
from Cryptodome.IO import PEM
from Cryptodome.Util.py3compat import b, tostr, FileNotFoundError
from Cryptodome.Util.number import inverse, bytes_to_long
from Cryptodome.Util import asn1
def der2pem(der, text='PUBLIC'):
    import binascii
    chunks = [binascii.b2a_base64(der[i:i + 48]) for i in range(0, len(der), 48)]
    pem = b('-----BEGIN %s KEY-----\n' % text)
    pem += b('').join(chunks)
    pem += b('-----END %s KEY-----' % text)
    return pem