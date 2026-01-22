from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def create_or_update_express_route(self, params):
    """
        Create or update Express route.
        :return: create or update Express route instance state dictionary
        """
    self.log('create or update Express Route {0}'.format(self.name))
    try:
        params['sku']['name'] = params.get('sku').get('tier') + '_' + params.get('sku').get('family')
        poller = self.network_client.express_route_circuits.begin_create_or_update(resource_group_name=params.get('resource_group'), circuit_name=params.get('name'), parameters=params)
        result = self.get_poller_result(poller)
        self.log('Response : {0}'.format(result))
    except Exception as ex:
        self.fail('Failed to create express route {0} in resource group {1}: {2}'.format(self.name, self.resource_group, str(ex)))
    return express_route_to_dict(result)