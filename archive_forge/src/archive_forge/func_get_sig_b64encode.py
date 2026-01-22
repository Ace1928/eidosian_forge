from __future__ import (absolute_import, division, print_function)
from base64 import b64encode
from email.utils import formatdate
import re
import json
import hashlib
from ansible.module_utils.six import iteritems
from ansible.module_utils.six.moves.urllib.parse import urlparse, urlencode
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.basic import env_fallback
def get_sig_b64encode(self, data):
    """
        Generates a signed digest from a String

        :param digest: string to be signed & hashed
        :return: instance of digest object
        """
    r = re.compile('\\s*-----BEGIN (.*)-----\\s+')
    m = r.match(self.private_key)
    if not m:
        raise ValueError('Not a valid PEM pre boundary')
    pem_header = m.group(1)
    key = serialization.load_pem_private_key(self.private_key.encode(), None, default_backend())
    if pem_header == 'RSA PRIVATE KEY':
        sign = key.sign(data.encode(), padding.PKCS1v15(), hashes.SHA256())
        self.digest_algorithm = 'rsa-sha256'
    elif pem_header == 'EC PRIVATE KEY':
        sign = key.sign(data.encode(), ec.ECDSA(hashes.SHA256()))
        self.digest_algorithm = 'hs2019'
    else:
        raise Exception('Unsupported key: {0}'.format(pem_header))
    return b64encode(sign)