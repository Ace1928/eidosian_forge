from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def update_storage_share(self, old_responce):
    """
        Method calling the Azure SDK to update storage file share.
        :param old_response: dict with properties of the storage file share
        :return: dict with description of the new storage file share
        """
    self.log('Creating file share {0}'.format(self.name))
    file_share_details = dict(access_tier=self.access_tier if self.access_tier else old_responce.get('access_tier'), share_quota=self.quota if self.quota else old_responce.get('share_quota'), metadata=self.metadata if self.metadata else old_responce.get('metadata'), enabled_protocols=self.enabled_protocols if self.enabled_protocols else old_responce.get('enabled_protocols'), root_squash=self.root_squash if self.root_squash else old_responce.get('self.root_squash'))
    try:
        self.storage_client.file_shares.update(resource_group_name=self.resource_group, account_name=self.account_name, share_name=self.name, file_share=file_share_details)
    except Exception as e:
        self.fail('Error updating file share {0} : {1}'.format(self.name, str(e)))
    return self.get_share()