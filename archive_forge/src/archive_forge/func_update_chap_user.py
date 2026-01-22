from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble import __version__ as NIMBLE_ANSIBLE_VERSION
import ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble as utils
def update_chap_user(client_obj, user_name, **kwargs):
    if utils.is_null_or_empty(user_name):
        return (False, False, 'Update chap user failed as user is not present.', {}, {})
    try:
        user_resp = client_obj.chap_users.get(id=None, name=user_name)
        if utils.is_null_or_empty(user_resp):
            return (False, False, f"Chap user '{user_name}' cannot be updated as it is not present.", {}, {})
        changed_attrs_dict, params = utils.remove_unchanged_or_null_args(user_resp, **kwargs)
        if changed_attrs_dict.__len__() > 0:
            user_resp = client_obj.chap_users.update(id=user_resp.attrs.get('id'), **params)
            return (True, True, f"Chap user '{user_name}' already present. Modified the following attributes '{changed_attrs_dict}'", changed_attrs_dict, user_resp.attrs)
        else:
            return (True, False, f"Chap user '{user_resp.attrs.get('name')}' already present in given state.", {}, user_resp.attrs)
    except Exception as ex:
        return (False, False, f'Chap user update failed |{ex}', {}, {})