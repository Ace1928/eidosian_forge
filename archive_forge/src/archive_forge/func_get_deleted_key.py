from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def get_deleted_key(self):
    """
        Gets the properties of the specified deleted key in key vault.

        :return: deserialized key state dictionary
        """
    self.log('Get the key {0}'.format(self.name))
    results = []
    try:
        response = self._client.get_deleted_key(name=self.name)
        if response:
            response = deletedkeybundle_to_dict(response)
            if self.has_tags(response['tags'], self.tags):
                self.log('Response : {0}'.format(response))
                results.append(response)
    except Exception as e:
        self.log('Did not find the key vault key {0}: {1}'.format(self.name, str(e)))
    return results