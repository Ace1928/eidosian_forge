from __future__ import absolute_import, division, print_function
import os
from base64 import b64encode, b64decode
from getpass import getuser
from socket import gethostname
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
def compare_publickeys(pk1, pk2):
    a = isinstance(pk1, Ed25519PublicKey)
    b = isinstance(pk2, Ed25519PublicKey)
    if a or b:
        if not a or not b:
            return False
        a = pk1.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        b = pk2.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        return a == b
    else:
        return pk1.public_numbers() == pk2.public_numbers()