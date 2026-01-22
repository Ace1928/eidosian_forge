from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, azure_id_to_dict
def delete_repository(self, repository_name):
    try:
        self._client.delete_repository(repository=repository_name)
    except Exception as e:
        self.fail(f'Could not delete repository {repository_name} - {str(e)}')