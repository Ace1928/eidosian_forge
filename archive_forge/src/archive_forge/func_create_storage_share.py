from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def create_storage_share(self):
    """
        Method calling the Azure SDK to create storage file share.
        :return: dict with description of the new storage file share
        """
    self.log('Creating fileshare {0}'.format(self.name))
    try:
        self.storage_client.file_shares.create(resource_group_name=self.resource_group, account_name=self.account_name, share_name=self.name, file_share=dict(access_tier=self.access_tier, share_quota=self.quota, metadata=self.metadata, root_squash=self.root_squash, enabled_protocols=self.enabled_protocols))
    except Exception as e:
        self.fail('Error creating file share {0} : {1}'.format(self.name, str(e)))
    return self.get_share()