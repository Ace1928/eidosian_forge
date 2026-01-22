from __future__ import absolute_import, division, print_function
import time
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def delete_replication(self):
    """
        Deletes specified Replication instance in the specified subscription and resource group.

        :return: True
        """
    self.log('Deleting the Replication instance {0}'.format(self.replication_name))
    try:
        response = self.containerregistry_client.replications.begin_delete(resource_group_name=self.resource_group, registry_name=self.registry_name, replication_name=self.replication_name)
        self.get_poller_result(response)
    except Exception as e:
        self.log('Error attempting to delete the Replication instance.')
        self.fail('Error deleting the Replication instance: {0}'.format(str(e)))
    return True