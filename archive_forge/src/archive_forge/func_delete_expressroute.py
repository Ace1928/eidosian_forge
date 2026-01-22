from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def delete_expressroute(self):
    """
        Deletes specified express route circuit
        :return True
        """
    self.log('Deleting the express route {0}'.format(self.name))
    try:
        poller = self.network_client.express_route_circuits.begin_delete(self.resource_group, self.name)
        result = self.get_poller_result(poller)
    except Exception as e:
        self.log('Error attempting to delete express route.')
        self.fail('Error deleting the express route : {0}'.format(str(e)))
    return result