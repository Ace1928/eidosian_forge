from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible_collections.purestorage.fusion.plugins.module_utils.networking import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
from ansible_collections.purestorage.fusion.plugins.module_utils.operations import (
def delete_se(module, fusion):
    """Delete Storage Endpoint"""
    se_api_instance = purefusion.StorageEndpointsApi(fusion)
    if not module.check_mode:
        op = se_api_instance.delete_storage_endpoint(region_name=module.params['region'], availability_zone_name=module.params['availability_zone'], storage_endpoint_name=module.params['name'])
        await_operation(fusion, op)
    module.exit_json(changed=True)