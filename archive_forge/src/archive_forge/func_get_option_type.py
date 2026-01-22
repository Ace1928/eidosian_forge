from __future__ import absolute_import, division, print_function
import abc
import binascii
import os
from base64 import b64encode
from datetime import datetime
from hashlib import sha256
from ansible.module_utils import six
from ansible.module_utils.common.text.converters import to_text
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import convert_relative_to_datetime
from ansible_collections.community.crypto.plugins.module_utils.openssh.utils import (
def get_option_type(name):
    if name in _CRITICAL_OPTIONS:
        result = 'critical'
    elif name in _EXTENSIONS:
        result = 'extension'
    else:
        raise ValueError('%s is not a valid option. ' % name + "Custom options must start with 'critical:' or 'extension:' to indicate type")
    return result