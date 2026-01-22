from __future__ import (absolute_import, division, print_function)
import errno
import fcntl
import os
import random
import shlex
import shutil
import subprocess
import sys
import tempfile
import warnings
from binascii import hexlify
from binascii import unhexlify
from binascii import Error as BinasciiError
from ansible.errors import AnsibleError, AnsibleAssertionError
from ansible import constants as C
from ansible.module_utils.six import binary_type
from ansible.module_utils.common.text.converters import to_bytes, to_text, to_native
from ansible.utils.display import Display
from ansible.utils.path import makedirs_safe, unfrackpath
def match_encrypt_secret(secrets, encrypt_vault_id=None):
    """Find the best/first/only secret in secrets to use for encrypting"""
    display.vvvv(u'encrypt_vault_id=%s' % to_text(encrypt_vault_id))
    if encrypt_vault_id:
        return match_encrypt_vault_id_secret(secrets, encrypt_vault_id=encrypt_vault_id)
    _vault_id_matchers = [_vault_id for _vault_id, dummy in secrets]
    best_secret = match_best_secret(secrets, _vault_id_matchers)
    return best_secret