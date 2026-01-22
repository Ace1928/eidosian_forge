from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
def create_update_resource_private_endpoint(self, privateendpoint):
    try:
        poller = self.network_client.private_endpoints.begin_create_or_update(resource_group_name=self.resource_group, private_endpoint_name=self.name, parameters=privateendpoint)
        new_privateendpoint = self.get_poller_result(poller)
    except Exception as exc:
        self.fail('Error creating or updating private endpoint {0} - {1}'.format(self.name, str(exc)))
    return self.private_endpoints_to_dict(new_privateendpoint)