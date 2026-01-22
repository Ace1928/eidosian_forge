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
def select_message_digest(digest_string):
    digest = None
    if digest_string == 'sha256':
        digest = hashes.SHA256()
    elif digest_string == 'sha384':
        digest = hashes.SHA384()
    elif digest_string == 'sha512':
        digest = hashes.SHA512()
    elif digest_string == 'sha1':
        digest = hashes.SHA1()
    elif digest_string == 'md5':
        digest = hashes.MD5()
    return digest