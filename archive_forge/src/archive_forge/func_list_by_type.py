from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import (
def list_by_type(self):
    """Get all Azure Traffic Managers endpoints of a profile by type"""
    self.log('List all Traffic Manager endpoints of a profile by type')
    try:
        response = self.traffic_manager_management_client.profiles.get(self.resource_group, self.profile_name)
    except Exception as exc:
        self.fail('Failed to list all items - {0}'.format(str(exc)))
    results = []
    for item in response:
        if item.endpoints:
            for endpoint in item.endpoints:
                if endpoint.type == self.type:
                    results.append(serialize_endpoint(endpoint, self.resource_group))
    return results