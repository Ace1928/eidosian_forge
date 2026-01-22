from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def get_deleted_secret(self):
    """
        Gets the properties of the specified deleted secret in key vault.

        :return: deserialized secret state dictionary
        """
    self.log('Get the secret {0}'.format(self.name))
    results = []
    try:
        response = self._client.get_deleted_secret(name=self.name)
        if response:
            response = deletedsecretbundle_to_dict(response)
            if self.has_tags(response['tags'], self.tags):
                self.log('Response : {0}'.format(response))
                results.append(response)
    except Exception as e:
        self.log('Did not find the key vault secret {0}: {1}'.format(self.name, str(e)))
    return results