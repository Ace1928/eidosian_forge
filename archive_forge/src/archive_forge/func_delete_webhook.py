from __future__ import absolute_import, division, print_function
import time
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def delete_webhook(self):
    """
        Deletes specified Webhook instance in the specified subscription and resource group.

        :return: True
        """
    self.log('Deleting the Webhook instance {0}'.format(self.webhook_name))
    try:
        response = self.containerregistry_client.webhooks.begin_delete(resource_group_name=self.resource_group, registry_name=self.registry_name, webhook_name=self.webhook_name)
        self.get_poller_result(response)
    except Exception as e:
        self.log('Error attempting to delete the Webhook instance.')
        self.fail('Error deleting the Webhook instance: {0}'.format(str(e)))
    return True