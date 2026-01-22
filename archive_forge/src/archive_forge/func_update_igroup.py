from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble import __version__ as NIMBLE_ANSIBLE_VERSION
import ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble as utils
def update_igroup(client_obj, ig_resp, **kwargs):
    if utils.is_null_or_empty(ig_resp):
        return (False, False, 'Update initiator group failed as it is not present.', {}, {})
    try:
        ig_name = ig_resp.attrs.get('name')
        changed_attrs_dict, params = utils.remove_unchanged_or_null_args(ig_resp, **kwargs)
        if changed_attrs_dict.__len__() > 0:
            ig_resp = client_obj.initiator_groups.update(id=ig_resp.attrs.get('id'), **params)
            return (True, True, f"Initiator group '{ig_name}' already present. Modified the following attributes '{changed_attrs_dict}'", changed_attrs_dict, ig_resp.attrs)
        else:
            return (True, False, f"Initiator group '{ig_name}' already present in given state.", {}, ig_resp.attrs)
    except Exception as ex:
        return (False, False, f'Initiator group update failed | {ex}', {}, {})