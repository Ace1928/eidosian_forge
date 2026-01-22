from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible_collections.purestorage.fusion.plugins.module_utils.networking import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
from ansible_collections.purestorage.fusion.plugins.module_utils.operations import (
def get_se(module, fusion):
    """Storage Endpoint or None"""
    se_api_instance = purefusion.StorageEndpointsApi(fusion)
    try:
        return se_api_instance.get_storage_endpoint(region_name=module.params['region'], storage_endpoint_name=module.params['name'], availability_zone_name=module.params['availability_zone'])
    except purefusion.rest.ApiException:
        return None