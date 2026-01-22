from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def get_traffic_manager_profile(self):
    """
        Gets the properties of the specified Traffic Manager profile

        :return: deserialized Traffic Manager profile dict
        """
    self.log('Checking if Traffic Manager profile {0} is present'.format(self.name))
    try:
        response = self.traffic_manager_management_client.profiles.get(self.resource_group, self.name)
        self.log('Response : {0}'.format(response))
        self.log('Traffic Manager profile : {0} found'.format(response.name))
        return traffic_manager_profile_to_dict(response)
    except ResourceNotFoundError:
        self.log('Did not find the Traffic Manager profile.')
        return False