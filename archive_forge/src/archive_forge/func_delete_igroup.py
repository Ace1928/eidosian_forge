from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble import __version__ as NIMBLE_ANSIBLE_VERSION
import ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble as utils
def delete_igroup(client_obj, initiator_group_name):
    if utils.is_null_or_empty(initiator_group_name):
        return (False, False, 'Initiator group deletion failed as it is not present.', {})
    try:
        ig_resp = client_obj.initiator_groups.get(id=None, name=initiator_group_name)
        if ig_resp is not None:
            client_obj.initiator_groups.delete(ig_resp.attrs.get('id'))
            return (True, True, f"Successfully deleted initiator group '{initiator_group_name}'.", {})
        elif ig_resp is None:
            return (False, False, f"Initiator group '{initiator_group_name}' is not present on array.", {})
        else:
            return (False, False, f"Failed to delete initiator group '{initiator_group_name}'.", {})
    except Exception as ex:
        return (False, False, f'Initiator group deletion failed | {ex}', {})