from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def create_update_webapp(self):
    """
        Creates or updates Web App with the specified configuration.

        :return: deserialized Web App instance state dictionary
        """
    self.log('Creating / Updating the Web App instance {0}'.format(self.name))
    try:
        response = self.web_client.web_apps.begin_create_or_update(resource_group_name=self.resource_group, name=self.name, site_envelope=self.site)
        if isinstance(response, LROPoller):
            response = self.get_poller_result(response)
    except Exception as exc:
        self.log('Error attempting to create the Web App instance.')
        self.fail('Error creating the Web App instance: {0}'.format(str(exc)))
    return webapp_to_dict(response)