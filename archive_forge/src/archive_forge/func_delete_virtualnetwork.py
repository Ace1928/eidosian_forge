from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def delete_virtualnetwork(self):
    """
        Deletes specified Virtual Network instance in the specified subscription and resource group.

        :return: True
        """
    self.log('Deleting the Virtual Network instance {0}'.format(self.name))
    try:
        response = self.mgmt_client.virtual_networks.begin_delete(resource_group_name=self.resource_group, lab_name=self.lab_name, name=self.name)
    except Exception as e:
        self.log('Error attempting to delete the Virtual Network instance.')
        self.fail('Error deleting the Virtual Network instance: {0}'.format(str(e)))
    return True