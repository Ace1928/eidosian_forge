from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def update_configuration_slot(self, slot_name=None, site_config=None):
    """
        Update slot configuration
        :return: deserialized slot configuration response
        """
    self.log('Update web app slot configuration')
    if slot_name is None:
        slot_name = self.name
    if site_config is None:
        site_config = self.site_config
    try:
        response = self.web_client.web_apps.update_configuration_slot(resource_group_name=self.resource_group, name=self.webapp_name, slot=slot_name, site_config=site_config)
        self.log('Response : {0}'.format(response))
        return response
    except Exception as ex:
        self.fail('Failed to update configuration for web app slot {0} in resource group {1}: {2}'.format(slot_name, self.resource_group, str(ex)))