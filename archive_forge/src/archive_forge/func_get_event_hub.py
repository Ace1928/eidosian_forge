from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def get_event_hub(self):
    """
        get event hub using resource_group, namespace_name and name.
        """
    response = None
    results = []
    try:
        response = self.event_hub_client.event_hubs.get(self.resource_group, self.namespace_name, self.name)
    except ResourceNotFoundError as e:
        self.fail('Could not get info for event hub. {0}').format(str(e))
    if response:
        results = [response]
    return results