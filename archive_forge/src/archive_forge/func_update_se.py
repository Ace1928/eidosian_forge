from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible_collections.purestorage.fusion.plugins.module_utils.networking import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
from ansible_collections.purestorage.fusion.plugins.module_utils.operations import (
def update_se(module, fusion, se):
    """Update Storage Endpoint"""
    se_api_instance = purefusion.StorageEndpointsApi(fusion)
    patches = []
    if module.params['display_name'] and module.params['display_name'] != se.display_name:
        patch = purefusion.StorageEndpointPatch(display_name=purefusion.NullableString(module.params['display_name']))
        patches.append(patch)
    if not module.check_mode:
        for patch in patches:
            op = se_api_instance.update_storage_endpoint(patch, region_name=module.params['region'], availability_zone_name=module.params['availability_zone'], storage_endpoint_name=module.params['name'])
            await_operation(fusion, op)
    changed = len(patches) != 0
    module.exit_json(changed=changed, id=se.id)