from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
import time
import http
@_api_permission_denied_handler('volumes')
def generate_volumes_dict(module, fusion):
    volume_info = {}
    tenant_api_instance = purefusion.TenantsApi(fusion)
    vol_api_instance = purefusion.VolumesApi(fusion)
    tenant_space_api_instance = purefusion.TenantSpacesApi(fusion)
    tenants = tenant_api_instance.list_tenants()
    for tenant in tenants.items:
        tenant_spaces = tenant_space_api_instance.list_tenant_spaces(tenant_name=tenant.name).items
        for tenant_space in tenant_spaces:
            volumes = vol_api_instance.list_volumes(tenant_name=tenant.name, tenant_space_name=tenant_space.name)
            for volume in volumes.items:
                vol_name = tenant.name + '/' + tenant_space.name + '/' + volume.name
                volume_info[vol_name] = {'tenant': tenant.name, 'tenant_space': tenant_space.name, 'name': volume.name, 'size': volume.size, 'display_name': volume.display_name, 'placement_group': volume.placement_group.name, 'source_volume_snapshot': getattr(volume.source_volume_snapshot, 'name', None), 'protection_policy': getattr(volume.protection_policy, 'name', None), 'storage_class': volume.storage_class.name, 'serial_number': volume.serial_number, 'target': {}, 'array': getattr(volume.array, 'name', None)}
                volume_info[vol_name]['target'] = {'iscsi': {'addresses': volume.target.iscsi.addresses, 'iqn': volume.target.iscsi.iqn}, 'nvme': {'addresses': None, 'nqn': None}, 'fc': {'addresses': None, 'wwns': None}}
    return volume_info