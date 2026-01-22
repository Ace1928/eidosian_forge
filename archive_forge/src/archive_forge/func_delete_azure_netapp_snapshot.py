from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import to_native
from ansible_collections.netapp.azure.plugins.module_utils.azure_rm_netapp_common import AzureRMNetAppModuleBase
from ansible_collections.netapp.azure.plugins.module_utils.netapp_module import NetAppModule
def delete_azure_netapp_snapshot(self):
    """
            Delete a snapshot for the given Azure NetApp Account
            :return: None
        """
    try:
        result = self.get_method('snapshots', 'delete')(resource_group_name=self.parameters['resource_group'], account_name=self.parameters['account_name'], pool_name=self.parameters['pool_name'], volume_name=self.parameters['volume_name'], snapshot_name=self.parameters['name'])
        while result.done() is not True:
            result.result(10)
    except (CloudError, AzureError) as error:
        self.module.fail_json(msg='Error deleting snapshot %s for Azure NetApp account %s: %s' % (self.parameters['name'], self.parameters['account_name'], to_native(error)), exception=traceback.format_exc())