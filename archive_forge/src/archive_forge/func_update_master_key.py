from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble import __version__ as NIMBLE_ANSIBLE_VERSION
import ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble as utils
def update_master_key(client_obj, master_key, **kwargs):
    if utils.is_null_or_empty(master_key):
        return (False, False, 'Update master key failed as master key is not present.', {}, {})
    try:
        master_key_resp = client_obj.master_key.get(id=None, name=master_key)
        if utils.is_null_or_empty(master_key_resp):
            return (False, False, f"Master key '{master_key}' cannot be updated as it is not present.", {}, {})
        changed_attrs_dict, params = utils.remove_unchanged_or_null_args(master_key_resp, **kwargs)
        changed_attrs_dict.pop('passphrase')
        if changed_attrs_dict.__len__() > 0:
            master_key_resp = client_obj.master_key.update(id=master_key_resp.attrs.get('id'), name=master_key, **params)
            return (True, True, f"Master key '{master_key}' already present. Modified the following attributes '{changed_attrs_dict}'", changed_attrs_dict, master_key_resp.attrs)
        else:
            return (True, False, f"Master key '{master_key}' already present in given state.", {}, master_key_resp.attrs)
    except Exception as ex:
        return (False, False, f'Master key update failed |{ex}', {}, {})