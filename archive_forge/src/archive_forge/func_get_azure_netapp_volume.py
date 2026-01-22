from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import to_native
from ansible_collections.netapp.azure.plugins.module_utils.azure_rm_netapp_common import AzureRMNetAppModuleBase
from ansible_collections.netapp.azure.plugins.module_utils.netapp_module import NetAppModule
def get_azure_netapp_volume(self):
    """
            Returns volume object for an existing volume
            Return None if volume does not exist
        """
    try:
        volume_get = self.netapp_client.volumes.get(self.parameters['resource_group'], self.parameters['account_name'], self.parameters['pool_name'], self.parameters['name'])
    except (CloudError, ResourceNotFoundError):
        return None
    return self.dict_from_volume_object(volume_get)