from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def set_webapp_state(self, appstate):
    """
        Start/stop/restart web app
        :return: deserialized updating response
        """
    try:
        if appstate == 'started':
            response = self.web_client.web_apps.start(resource_group_name=self.resource_group, name=self.name)
        elif appstate == 'stopped':
            response = self.web_client.web_apps.stop(resource_group_name=self.resource_group, name=self.name)
        elif appstate == 'restarted':
            response = self.web_client.web_apps.restart(resource_group_name=self.resource_group, name=self.name)
        else:
            self.fail('Invalid web app state {0}'.format(appstate))
        self.log('Response : {0}'.format(response))
        return response
    except Exception as ex:
        request_id = ex.request_id if ex.request_id else ''
        self.log('Failed to {0} web app {1} in resource group {2}, request_id {3} - {4}'.format(appstate, self.name, self.resource_group, request_id, str(ex)))