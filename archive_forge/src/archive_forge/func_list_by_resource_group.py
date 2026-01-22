from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _camel_to_snake
def list_by_resource_group(self):
    response = None
    results = []
    try:
        response = self.containerinstance_client.container_groups.list_by_resource_group(resource_group_name=self.resource_group)
        self.log('Response : {0}'.format(response))
    except Exception as e:
        self.fail('Could not list facts for Container Instances.')
    if response is not None:
        for item in response:
            if self.has_tags(item.tags, self.tags):
                results.append(self.format_item(item))
    return results