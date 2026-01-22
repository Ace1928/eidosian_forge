from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
def create_update_zone(self):
    try:
        self.parameters['name'] = self.name
        response = self.network_client.private_dns_zone_groups.begin_create_or_update(resource_group_name=self.resource_group, private_endpoint_name=self.private_endpoint, private_dns_zone_group_name=self.name, parameters=self.parameters)
        if isinstance(response, LROPoller):
            response = self.get_poller_result(response)
        return self.zone_to_dict(response)
    except Exception as exc:
        self.fail('Error creating or updating DNS zone group {0} for private endpoint {1}: {2}'.format(self.name, self.private_endpoint, str(exc)))