from __future__ import absolute_import, division, print_function
import time
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def get_keyvault(self):
    """
        Gets the properties of the specified Key Vault.

        :return: deserialized Key Vault instance state dictionary
        """
    self.log('Checking if the Key Vault instance {0} is present'.format(self.vault_name))
    found = False
    try:
        response = self.mgmt_client.vaults.get(resource_group_name=self.resource_group, vault_name=self.vault_name)
        found = True
        self.log('Response : {0}'.format(response))
        self.log('Key Vault instance : {0} found'.format(response.name))
    except ResourceNotFoundError as e:
        self.log('Did not find the Key Vault instance.')
    if found is True:
        return response.as_dict()
    return False