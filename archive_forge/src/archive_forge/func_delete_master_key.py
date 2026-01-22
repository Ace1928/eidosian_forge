from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble import __version__ as NIMBLE_ANSIBLE_VERSION
import ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble as utils
def delete_master_key(client_obj, master_key):
    if utils.is_null_or_empty(master_key):
        return (False, False, 'Delete master key failed as master key is not present.', {})
    try:
        master_key_resp = client_obj.master_key.get(id=None, name=master_key)
        if utils.is_null_or_empty(master_key_resp):
            return (False, False, f"Master key '{master_key}' cannot be deleted as it is not present.", {})
        client_obj.master_key.delete(id=master_key_resp.attrs.get('id'))
        return (True, True, f"Deleted master key '{master_key}' successfully.", {})
    except Exception as ex:
        return (False, False, f'Delete master key failed |{ex}', {})