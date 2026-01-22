from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _snake_to_camel
def purge_cdnendpoint(self):
    """
        Purges an existing Azure CDN endpoint.

        :return: deserialized Azure CDN endpoint state dictionary
        """
    self.log('Purging the Azure CDN endpoint {0}'.format(self.name))
    try:
        poller = self.cdn_client.endpoints.begin_purge_content(self.resource_group, self.profile_name, self.name, content_file_paths=dict(content_paths=self.purge_content_paths))
        response = self.get_poller_result(poller)
        self.log('Response : {0}'.format(response))
        return self.get_cdnendpoint()
    except Exception as e:
        self.fail('Fail to purge the Azure CDN endpoint.')
        return False