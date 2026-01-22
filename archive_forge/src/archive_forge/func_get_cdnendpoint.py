from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _snake_to_camel
def get_cdnendpoint(self):
    """
        Gets the properties of the specified Azure CDN endpoint.

        :return: deserialized Azure CDN endpoint state dictionary
        """
    self.log('Checking if the Azure CDN endpoint {0} is present'.format(self.name))
    try:
        response = self.cdn_client.endpoints.get(self.resource_group, self.profile_name, self.name)
        self.log('Response : {0}'.format(response))
        self.log('Azure CDN endpoint : {0} found'.format(response.name))
        return cdnendpoint_to_dict(response)
    except Exception:
        self.log('Did not find the Azure CDN endpoint.')
        return False