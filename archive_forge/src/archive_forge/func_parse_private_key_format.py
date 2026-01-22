from __future__ import absolute_import, division, print_function
import abc
import os
import stat
import traceback
from ansible.module_utils import six
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.crypto.plugins.module_utils.openssh.utils import (
def parse_private_key_format(path):
    with open(path, 'r') as file:
        header = file.readline().strip()
    if header == '-----BEGIN OPENSSH PRIVATE KEY-----':
        return 'SSH'
    elif header == '-----BEGIN PRIVATE KEY-----':
        return 'PKCS8'
    elif header == '-----BEGIN RSA PRIVATE KEY-----':
        return 'PKCS1'
    return ''