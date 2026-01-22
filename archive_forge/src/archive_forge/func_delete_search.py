from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def delete_search(self):
    self.log('Delete search {0}'.format(self.name))
    try:
        if self.account_dict is not None:
            self.results['changed'] = True
            self.search_client.services.delete(self.resource_group, self.name)
    except Exception as e:
        self.fail('Failed to delete the search: {0}'.format(str(e)))