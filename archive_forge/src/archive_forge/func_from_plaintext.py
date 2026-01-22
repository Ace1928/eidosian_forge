from __future__ import (absolute_import, division, print_function)
import sys as _sys
from collections.abc import Sequence
from ansible.module_utils.six import text_type
from ansible.module_utils.common.text.converters import to_bytes, to_text, to_native
@classmethod
def from_plaintext(cls, seq, vault, secret):
    if not vault:
        raise vault.AnsibleVaultError('Error creating AnsibleVaultEncryptedUnicode, invalid vault (%s) provided' % vault)
    ciphertext = vault.encrypt(seq, secret)
    avu = cls(ciphertext)
    avu.vault = vault
    return avu