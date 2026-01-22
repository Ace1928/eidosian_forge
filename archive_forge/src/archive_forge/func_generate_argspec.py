from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.hashi_vault.plugins.module_utils._hashi_vault_common import (
from ansible_collections.community.hashi_vault.plugins.module_utils._connection_options import HashiVaultConnectionOptions
from ansible_collections.community.hashi_vault.plugins.module_utils._authenticator import HashiVaultAuthenticator
@classmethod
def generate_argspec(cls, **kwargs):
    spec = HashiVaultConnectionOptions.ARGSPEC.copy()
    spec.update(HashiVaultAuthenticator.ARGSPEC.copy())
    spec.update(**kwargs)
    return spec