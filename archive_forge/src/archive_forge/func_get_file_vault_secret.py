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
def get_file_vault_secret(filename=None, vault_id=None, encoding=None, loader=None):
    """ Get secret from file content or execute file and get secret from stdout """
    this_path = unfrackpath(filename, follow=False)
    if not os.path.exists(this_path):
        raise AnsibleError('The vault password file %s was not found' % this_path)
    if loader.is_executable(this_path):
        if script_is_client(filename):
            display.vvvv(u'The vault password file %s is a client script.' % to_text(this_path))
            return ClientScriptVaultSecret(filename=this_path, vault_id=vault_id, encoding=encoding, loader=loader)
        return ScriptVaultSecret(filename=this_path, encoding=encoding, loader=loader)
    return FileVaultSecret(filename=this_path, encoding=encoding, loader=loader)