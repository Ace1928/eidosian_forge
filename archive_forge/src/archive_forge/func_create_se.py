from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible_collections.purestorage.fusion.plugins.module_utils.networking import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
from ansible_collections.purestorage.fusion.plugins.module_utils.operations import (
def create_se(module, fusion):
    """Create Storage Endpoint"""
    se_api_instance = purefusion.StorageEndpointsApi(fusion)
    id = None
    if not module.check_mode:
        endpoint_type = None
        iscsi = None
        if module.params['iscsi'] is not None:
            iscsi = purefusion.StorageEndpointIscsiPost(discovery_interfaces=[purefusion.StorageEndpointIscsiDiscoveryInterfacePost(**endpoint) for endpoint in module.params['iscsi']])
            endpoint_type = 'iscsi'
        cbs_azure_iscsi = None
        if module.params['cbs_azure_iscsi'] is not None:
            cbs_azure_iscsi = purefusion.StorageEndpointCbsAzureIscsiPost(storage_endpoint_collection_identity=module.params['cbs_azure_iscsi']['storage_endpoint_collection_identity'], load_balancer=module.params['cbs_azure_iscsi']['load_balancer'], load_balancer_addresses=module.params['cbs_azure_iscsi']['load_balancer_addresses'])
            endpoint_type = 'cbs-azure-iscsi'
        op = se_api_instance.create_storage_endpoint(purefusion.StorageEndpointPost(name=module.params['name'], display_name=module.params['display_name'] or module.params['name'], endpoint_type=endpoint_type, iscsi=iscsi, cbs_azure_iscsi=cbs_azure_iscsi), region_name=module.params['region'], availability_zone_name=module.params['availability_zone'])
        res_op = await_operation(fusion, op)
        id = res_op.result.resource.id
    module.exit_json(changed=True, id=id)