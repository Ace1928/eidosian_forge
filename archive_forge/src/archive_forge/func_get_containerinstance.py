from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _snake_to_camel
def get_containerinstance(self):
    """
        Gets the properties of the specified container service.

        :return: deserialized container instance state dictionary
        """
    self.log('Checking if the container instance {0} is present'.format(self.name))
    found = False
    try:
        response = self.containerinstance_client.container_groups.get(resource_group_name=self.resource_group, container_group_name=self.name)
        found = True
        self.log('Response : {0}'.format(response))
        self.log('Container instance : {0} found'.format(response.name))
    except ResourceNotFoundError as e:
        self.log('Did not find the container instance.')
    if found is True:
        return response.as_dict()
    return False