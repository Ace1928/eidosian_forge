from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
import time
def create_or_update_event_hub(self):
    """
        Create or update Event Hub.
        :return: create or update Event Hub instance state dictionary
        """
    try:
        if self.sku == 'Basic':
            self.message_retention_in_days = 1
        params = Eventhub(message_retention_in_days=self.message_retention_in_days, partition_count=self.partition_count, status=self.status)
        result = self.event_hub_client.event_hubs.create_or_update(self.resource_group, self.namespace_name, self.name, params)
        self.log('Response : {0}'.format(result))
    except Exception as ex:
        self.fail('Failed to create event hub {0} in resource group {1}: {2}'.format(self.name, self.resource_group, str(ex)))
    return event_hub_to_dict(result)