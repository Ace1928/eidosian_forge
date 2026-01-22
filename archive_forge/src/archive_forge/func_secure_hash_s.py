from __future__ import (absolute_import, division, print_function)
import os
from hashlib import sha1
from ansible.errors import AnsibleError
from ansible.module_utils.common.text.converters import to_bytes
def secure_hash_s(data, hash_func=sha1):
    """ Return a secure hash hex digest of data. """
    digest = hash_func()
    data = to_bytes(data, errors='surrogate_or_strict')
    digest.update(data)
    return digest.hexdigest()