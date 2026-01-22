from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def get_webapp(self):
    """
        Gets the properties of the specified Web App.

        :return: deserialized Web App instance state dictionary
        """
    self.log('Checking if the Web App instance {0} is present'.format(self.name))
    response = None
    try:
        response = self.web_client.web_apps.get(resource_group_name=self.resource_group, name=self.name)
        if response is not None:
            self.log('Response : {0}'.format(response))
            self.log('Web App instance : {0} found'.format(response.name))
            return webapp_to_dict(response)
    except ResourceNotFoundError:
        pass
    self.log("Didn't find web app {0} in resource group {1}".format(self.name, self.resource_group))
    return False