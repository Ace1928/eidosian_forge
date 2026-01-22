from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _snake_to_camel
def stop_cdnendpoint(self):
    """
        Stops an existing Azure CDN endpoint that is on a running state.

        :return: deserialized Azure CDN endpoint state dictionary
        """
    self.log('Stopping the Azure CDN endpoint {0}'.format(self.name))
    try:
        poller = self.cdn_client.endpoints.begin_stop(self.resource_group, self.profile_name, self.name)
        response = self.get_poller_result(poller)
        self.log('Response : {0}'.format(response))
        self.log('Azure CDN endpoint : {0} stopped'.format(response.name))
        return self.get_cdnendpoint()
    except Exception:
        self.log('Fail to stop the Azure CDN endpoint.')
        return False