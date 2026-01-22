from __future__ import absolute_import, division, print_function
import time
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def delete_notification_hub(self):
    """
        Deletes specified notication hub
        :return True
        """
    self.log('Deleting the notification hub {0}'.format(self.name))
    try:
        result = self.notification_hub_client.notification_hubs.delete(self.resource_group, self.namespace_name, self.name)
    except Exception as e:
        self.log('Error attempting to delete notification hub.')
        self.fail('Error deleting the notification hub : {0}'.format(str(e)))
    return True