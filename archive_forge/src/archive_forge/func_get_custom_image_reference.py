from __future__ import absolute_import, division, print_function
import base64
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, azure_id_to_dict, format_resource_id
from ansible.module_utils.basic import to_native, to_bytes
def get_custom_image_reference(self, name, resource_group=None):
    try:
        if resource_group:
            vm_images = self.compute_client.images.list_by_resource_group(resource_group)
        else:
            vm_images = self.compute_client.images.list()
    except ResourceNotFoundError as exc:
        self.fail('Error fetching custom images from subscription - {0}'.format(str(exc)))
    for vm_image in vm_images:
        if vm_image.name == name:
            self.log('Using custom image id {0}'.format(vm_image.id))
            return self.compute_models.ImageReference(id=vm_image.id)
    self.fail('Error could not find image with name {0}'.format(name))