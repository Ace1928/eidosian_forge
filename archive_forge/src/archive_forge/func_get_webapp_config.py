from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def get_webapp_config(self):
    try:
        return self.web_client.web_apps.get_configuration(resource_group_name=self.resource_group, name=self.name)
    except Exception as exc:
        self.fail('Error getting webapp config {0} (rg={1}) - {2}'.format(self.name, self.resource_group, str(exc)))