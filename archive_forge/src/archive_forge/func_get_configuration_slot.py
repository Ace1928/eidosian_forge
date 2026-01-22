from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def get_configuration_slot(self, slot_name):
    """
        Get slot configuration
        :return: deserialized slot configuration response
        """
    self.log('Get web app slot configuration')
    try:
        response = self.web_client.web_apps.get_configuration_slot(resource_group_name=self.resource_group, name=self.webapp_name, slot=slot_name)
        self.log('Response : {0}'.format(response))
        return response
    except Exception as ex:
        self.fail('Failed to get configuration for web app slot {0} in resource group {1}: {2}'.format(slot_name, self.resource_group, str(ex)))