from __future__ import (absolute_import, division, print_function)
from jinja2.runtime import Undefined
from jinja2.exceptions import UndefinedError
from ansible.errors import AnsibleFilterError, AnsibleFilterTypeError
from ansible.module_utils.common.text.converters import to_native, to_bytes
from ansible.module_utils.six import string_types, binary_type
from ansible.parsing.yaml.objects import AnsibleVaultEncryptedUnicode
from ansible.parsing.vault import is_encrypted, VaultSecret, VaultLib
from ansible.utils.display import Display
def do_vault(data, secret, salt=None, vault_id='filter_default', wrap_object=False, vaultid=None):
    if not isinstance(secret, (string_types, binary_type, Undefined)):
        raise AnsibleFilterTypeError('Secret passed is required to be a string, instead we got: %s' % type(secret))
    if not isinstance(data, (string_types, binary_type, Undefined)):
        raise AnsibleFilterTypeError('Can only vault strings, instead we got: %s' % type(data))
    if vaultid is not None:
        display.deprecated("Use of undocumented 'vaultid', use 'vault_id' instead", version='2.20')
        if vault_id == 'filter_default':
            vault_id = vaultid
        else:
            display.warning('Ignoring vaultid as vault_id is already set.')
    vault = ''
    vs = VaultSecret(to_bytes(secret))
    vl = VaultLib()
    try:
        vault = vl.encrypt(to_bytes(data), vs, vault_id, salt)
    except UndefinedError:
        raise
    except Exception as e:
        raise AnsibleFilterError('Unable to encrypt: %s' % to_native(e), orig_exc=e)
    if wrap_object:
        vault = AnsibleVaultEncryptedUnicode(vault)
    else:
        vault = to_native(vault)
    return vault