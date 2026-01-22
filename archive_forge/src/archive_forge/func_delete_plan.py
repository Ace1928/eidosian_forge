from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def delete_plan(self):
    """
        Deletes specified App service plan in the specified subscription and resource group.

        :return: True
        """
    self.log('Deleting the App service plan {0}'.format(self.name))
    try:
        self.web_client.app_service_plans.delete(resource_group_name=self.resource_group, name=self.name)
    except ResourceNotFoundError as e:
        self.log('Error attempting to delete App service plan.')
        self.fail('Error deleting the App service plan : {0}'.format(str(e)))
    return True