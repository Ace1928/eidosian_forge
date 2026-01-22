from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def list_by_repository(self, repository_name, tag_name):
    try:
        response = self._client.list_tag_properties(repository=repository_name)
        self.log(f'Response : {response}')
        tags = []
        for tag in response:
            if not tag_name or tag.name == tag_name:
                tags.append(format_tag(tag))
        return {'name': repository_name, 'tags': tags}
    except ResourceNotFoundError as e:
        self.log(f'Could not get ACR tags for {repository_name} - {str(e)}')
    return None