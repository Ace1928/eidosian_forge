from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def get_search(self):
    self.log('Get properties for azure search {0}'.format(self.name))
    search_obj = None
    account_dict = None
    try:
        search_obj = self.search_client.services.get(self.resource_group, self.name)
    except ResourceNotFoundError:
        pass
    if search_obj:
        account_dict = self.account_obj_to_dict(search_obj)
    return account_dict