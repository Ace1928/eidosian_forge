from __future__ import absolute_import, division, print_function
import abc
import datetime
import errno
import hashlib
import os
import re
from ansible.module_utils import six
from ansible.module_utils.common.text.converters import to_native, to_bytes
from ansible_collections.community.crypto.plugins.module_utils.crypto.pem import (
from .basic import (
def get_fingerprint_of_privatekey(privatekey, backend='cryptography', prefer_one=False):
    """Generate the fingerprint of the public key. """
    if backend == 'cryptography':
        publickey = privatekey.public_key().public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
    return get_fingerprint_of_bytes(publickey, prefer_one=prefer_one)