from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_rest import GenericRestClient
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
import json
def listbytags(self):
    self.url = self.get_url_bytags()
    response = None
    try:
        response = self.mgmt_client.query(self.url, 'GET', self.query_parameters, self.header_parameters, None, self.status_code, 600, 30)
    except Exception as e:
        self.log('Could not get info for the given api tags {0}'.format(e))
    try:
        response = json.loads(response.body())
    except Exception:
        return None
    return response