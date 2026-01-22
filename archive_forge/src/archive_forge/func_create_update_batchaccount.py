from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
from ansible.module_utils.common.dict_transformations import _snake_to_camel
def create_update_batchaccount(self):
    """
        Creates or updates Batch Account with the specified configuration.

        :return: deserialized Batch Account instance state dictionary
        """
    self.log('Creating / Updating the Batch Account instance {0}'.format(self.name))
    try:
        if self.to_do == Actions.Create:
            response = self.mgmt_client.batch_account.begin_create(resource_group_name=self.resource_group, account_name=self.name, parameters=self.batch_account)
        else:
            response = self.mgmt_client.batch_account.update(resource_group_name=self.resource_group, account_name=self.name, parameters=dict(tags=self.tags, auto_storage=self.batch_account.get('self.batch_account')))
        if isinstance(response, LROPoller):
            response = self.get_poller_result(response)
    except Exception as exc:
        self.log('Error attempting to create the Batch Account instance.')
        self.fail('Error creating the Batch Account instance: {0}'.format(str(exc)))
    return response.as_dict()