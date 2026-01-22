from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble import __version__ as NIMBLE_ANSIBLE_VERSION
import ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble as utils
def create_master_key(client_obj, master_key, passphrase):
    if utils.is_null_or_empty(master_key):
        return (False, False, 'Create master key failed as no key is provided.', {}, {})
    try:
        master_key_resp = client_obj.master_key.get(id=None, name=master_key)
        if utils.is_null_or_empty(master_key_resp):
            master_key_resp = client_obj.master_key.create(name=master_key, passphrase=passphrase)
            return (True, True, f"Master key '{master_key}' created successfully.", {}, master_key_resp.attrs)
        else:
            return (False, False, f"Master key '{master_key}' cannot be created as it is already present in given state.", {}, master_key_resp.attrs)
    except Exception as ex:
        return (False, False, f'Master key creation failed |{ex}', {}, {})