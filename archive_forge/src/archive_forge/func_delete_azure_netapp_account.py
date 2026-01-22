from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import to_native
from ansible_collections.netapp.azure.plugins.module_utils.azure_rm_netapp_common import AzureRMNetAppModuleBase
from ansible_collections.netapp.azure.plugins.module_utils.netapp_module import NetAppModule
def delete_azure_netapp_account(self):
    """
            Delete an Azure NetApp Account
            :return: None
        """
    try:
        response = self.get_method('accounts', 'delete')(resource_group_name=self.parameters['resource_group'], account_name=self.parameters['name'])
        while response.done() is not True:
            response.result(10)
    except (CloudError, AzureError) as error:
        self.module.fail_json(msg='Error deleting Azure NetApp account %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())