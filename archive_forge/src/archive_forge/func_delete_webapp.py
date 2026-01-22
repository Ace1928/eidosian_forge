from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def delete_webapp(self):
    """
        Deletes specified Web App instance in the specified subscription and resource group.

        :return: True
        """
    self.log('Deleting the Web App instance {0}'.format(self.name))
    try:
        self.web_client.web_apps.delete(resource_group_name=self.resource_group, name=self.name)
    except Exception as e:
        self.log('Error attempting to delete the Web App instance.')
        self.fail('Error deleting the Web App instance: {0}'.format(str(e)))
    return True