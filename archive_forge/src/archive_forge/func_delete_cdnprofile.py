from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
import uuid
def delete_cdnprofile(self):
    """
        Deletes the specified Azure CDN profile in the specified subscription and resource group.

        :return: True
        """
    self.log('Deleting the CDN profile {0}'.format(self.name))
    try:
        poller = self.cdn_client.profiles.begin_delete(self.resource_group, self.name)
        self.get_poller_result(poller)
        return True
    except Exception as e:
        self.log('Error attempting to delete the CDN profile.')
        self.fail('Error deleting the CDN profile: {0}'.format(e.message))
        return False