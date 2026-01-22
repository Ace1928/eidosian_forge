from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble import __version__ as NIMBLE_ANSIBLE_VERSION
import ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble as utils
def delete_prot_schedule(client_obj, prot_schedule_name, volcoll_or_prottmpl_type, volcoll_or_prottmpl_id):
    if utils.is_null_or_empty(prot_schedule_name):
        return (False, False, 'Protection schedule deletion failed as protection schedule name is not present', {})
    try:
        prot_schedule_resp = client_obj.protection_schedules.get(id=None, name=prot_schedule_name, volcoll_or_prottmpl_type=volcoll_or_prottmpl_type, volcoll_or_prottmpl_id=volcoll_or_prottmpl_id)
        if utils.is_null_or_empty(prot_schedule_resp):
            return (False, False, f"Protection schedule '{prot_schedule_name}' not present to delete.", {})
        else:
            client_obj.protection_schedules.delete(id=prot_schedule_resp.attrs.get('id'))
            return (True, True, f"Deleted protection schedule '{prot_schedule_name}' successfully.", {})
    except Exception as ex:
        return (False, False, f'Protection schedule deletion failed | {ex}', {})