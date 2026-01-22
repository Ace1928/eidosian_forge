from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
from ansible.module_utils.common.dict_transformations import _snake_to_camel
def delete_batchaccount(self):
    """
        Deletes specified Batch Account instance in the specified subscription and resource group.

        :return: True
        """
    self.log('Deleting the Batch Account instance {0}'.format(self.name))
    try:
        response = self.mgmt_client.batch_account.begin_delete(resource_group_name=self.resource_group, account_name=self.name)
    except Exception as e:
        self.log('Error attempting to delete the Batch Account instance.')
        self.fail('Error deleting the Batch Account instance: {0}'.format(str(e)))
    if isinstance(response, LROPoller):
        response = self.get_poller_result(response)
    return True