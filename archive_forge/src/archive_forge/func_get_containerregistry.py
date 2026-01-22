from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def get_containerregistry(self):
    """
        Gets the properties of the specified container registry.

        :return: deserialized container registry state dictionary
        """
    self.log('Checking if the container registry instance {0} is present'.format(self.name))
    found = False
    try:
        response = self.containerregistry_client.registries.get(resource_group_name=self.resource_group, registry_name=self.name)
        found = True
        self.log('Response : {0}'.format(response))
        self.log('Container registry instance : {0} found'.format(response.name))
    except ResourceNotFoundError as e:
        self.log('Did not find the container registry instance: {0}'.format(str(e)))
        response = None
    if found is True and self.admin_user_enabled is True:
        try:
            credentials = self.containerregistry_client.registries.list_credentials(resource_group_name=self.resource_group, registry_name=self.name)
        except Exception as e:
            self.fail('List registry credentials failed: {0}'.format(str(e)))
            credentials = None
    elif found is True and self.admin_user_enabled is False:
        credentials = None
    else:
        return None
    return create_containerregistry_dict(response, credentials)