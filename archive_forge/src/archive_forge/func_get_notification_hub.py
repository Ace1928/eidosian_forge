from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def get_notification_hub(self):
    response = None
    results = []
    try:
        response = self.notification_hub_client.notification_hubs.get(self.resource_group, self.namespace_name, self.name)
        self.log('Response : {0}'.format(response))
    except ResourceNotFoundError as e:
        self.fail('Could not get info for notification hub. {0}').format(str(e))
    if response and self.has_tags(response.tags, self.tags):
        results = [response]
    return results