from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble import __version__ as NIMBLE_ANSIBLE_VERSION
import ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble as utils
def update_prot_schedule(client_obj, prot_schedule_resp, **kwargs):
    if utils.is_null_or_empty(prot_schedule_resp):
        return (False, False, 'Update protection schedule failed as protection schedule is not present.', {}, {})
    try:
        prot_schedule_name = prot_schedule_resp.attrs.get('name')
        changed_attrs_dict, params = utils.remove_unchanged_or_null_args(prot_schedule_resp, **kwargs)
        if changed_attrs_dict.__len__() > 0:
            prot_schedule_resp = client_obj.protection_schedules.update(id=prot_schedule_resp.attrs.get('id'), **params)
            return (True, True, f"Protection schedule '{prot_schedule_name}' already present. Modified the following attributes '{changed_attrs_dict}'", changed_attrs_dict, prot_schedule_resp.attrs)
        else:
            return (True, False, f"Protection schedule '{prot_schedule_name}' already present in given state.", {}, prot_schedule_resp.attrs)
    except Exception as ex:
        return (False, False, f'Protection schedule update failed |{ex}', {}, {})