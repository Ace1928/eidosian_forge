from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def delete_datalake_store(self):
    self.log('Delete datalake store {0}'.format(self.name))
    self.results['changed'] = True if self.account_dict is not None else False
    if not self.check_mode and self.account_dict is not None:
        try:
            status = self.datalake_store_client.accounts.begin_delete(self.resource_group, self.name)
            self.log('delete status: ')
            self.log(str(status))
        except Exception as e:
            self.fail('Failed to delete datalake store: {0}'.format(str(e)))
    return True