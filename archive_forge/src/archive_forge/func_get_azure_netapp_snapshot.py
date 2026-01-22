from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import to_native
from ansible_collections.netapp.azure.plugins.module_utils.azure_rm_netapp_common import AzureRMNetAppModuleBase
from ansible_collections.netapp.azure.plugins.module_utils.netapp_module import NetAppModule
def get_azure_netapp_snapshot(self):
    """
            Returns snapshot object for an existing snapshot
            Return None if snapshot does not exist
        """
    try:
        snapshot_get = self.netapp_client.snapshots.get(self.parameters['resource_group'], self.parameters['account_name'], self.parameters['pool_name'], self.parameters['volume_name'], self.parameters['name'])
    except (CloudError, ResourceNotFoundError):
        return None
    return snapshot_get