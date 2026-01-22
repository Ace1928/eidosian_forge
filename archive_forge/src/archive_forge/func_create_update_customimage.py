from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _snake_to_camel
def create_update_customimage(self):
    """
        Creates or updates Custom Image with the specified configuration.

        :return: deserialized Custom Image instance state dictionary
        """
    self.log('Creating / Updating the Custom Image instance {0}'.format(self.name))
    try:
        response = self.mgmt_client.custom_images.begin_create_or_update(resource_group_name=self.resource_group, lab_name=self.lab_name, name=self.name, custom_image=self.custom_image)
        if isinstance(response, LROPoller):
            response = self.get_poller_result(response)
    except Exception as exc:
        self.log('Error attempting to create the Custom Image instance.')
        self.fail('Error creating the Custom Image instance: {0}'.format(str(exc)))
    return response.as_dict()