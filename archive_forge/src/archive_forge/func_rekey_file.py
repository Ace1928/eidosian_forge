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
def rekey_file(self, filename, new_vault_secret, new_vault_id=None):
    filename = self._real_path(filename)
    prev = os.stat(filename)
    b_vaulttext = self.read_data(filename)
    vaulttext = to_text(b_vaulttext)
    display.vvvvv(u'Rekeying file "%s" to with new vault-id "%s" and vault secret %s' % (to_text(filename), to_text(new_vault_id), to_text(new_vault_secret)))
    try:
        plaintext, vault_id_used, _dummy = self.vault.decrypt_and_get_vault_id(vaulttext)
    except AnsibleError as e:
        raise AnsibleError('%s for %s' % (to_native(e), to_native(filename)))
    if new_vault_secret is None:
        raise AnsibleError('The value for the new_password to rekey %s with is not valid' % filename)
    new_vault = VaultLib(secrets={})
    b_new_vaulttext = new_vault.encrypt(plaintext, new_vault_secret, vault_id=new_vault_id)
    self.write_data(b_new_vaulttext, filename)
    os.chmod(filename, prev.st_mode)
    os.chown(filename, prev.st_uid, prev.st_gid)
    display.vvvvv(u'Rekeyed file "%s" (decrypted with vault id "%s") was encrypted with new vault-id "%s" and vault secret %s' % (to_text(filename), to_text(vault_id_used), to_text(new_vault_id), to_text(new_vault_secret)))