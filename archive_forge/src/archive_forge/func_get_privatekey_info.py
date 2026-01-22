from __future__ import absolute_import, division, print_function
import abc
import traceback
from ansible.module_utils import six
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_native, to_bytes
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.crypto.basic import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.math import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.publickey_info import (
def get_privatekey_info(module, backend, content, passphrase=None, return_private_key_data=False, prefer_one_fingerprint=False):
    if backend == 'cryptography':
        info = PrivateKeyInfoRetrievalCryptography(module, content, passphrase=passphrase, return_private_key_data=return_private_key_data)
    return info.get_info(prefer_one_fingerprint=prefer_one_fingerprint)