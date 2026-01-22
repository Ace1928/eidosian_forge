from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def list_images_by_resource_group(self, resource_group):
    """
        Returns image details based on its resource group
        """
    self.log('List images filtered by resource group')
    response = None
    try:
        response = self.image_client.images.list_by_resource_group(resource_group)
    except ResourceNotFoundError as exc:
        self.fail('Failed to list images: {0}'.format(str(exc)))
    return [self.format_item(x) for x in response if self.has_tags(x.tags, self.tags)] if response else []