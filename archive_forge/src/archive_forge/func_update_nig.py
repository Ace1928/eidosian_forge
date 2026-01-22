from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible_collections.purestorage.fusion.plugins.module_utils.networking import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
from ansible_collections.purestorage.fusion.plugins.module_utils.operations import (
def update_nig(module, fusion, nig):
    """Update Network Interface Group"""
    nifg_api_instance = purefusion.NetworkInterfaceGroupsApi(fusion)
    patches = []
    if module.params['display_name'] and module.params['display_name'] != nig.display_name:
        patch = purefusion.NetworkInterfaceGroupPatch(display_name=purefusion.NullableString(module.params['display_name']))
        patches.append(patch)
    if not module.check_mode:
        for patch in patches:
            op = nifg_api_instance.update_network_interface_group(patch, availability_zone_name=module.params['availability_zone'], region_name=module.params['region'], network_interface_group_name=module.params['name'])
            await_operation(fusion, op)
    changed = len(patches) != 0
    module.exit_json(changed=changed, id=nig.id)