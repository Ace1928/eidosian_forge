from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_rest import GenericRestClient
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
import json
def stop_protection_and_delete_data(self):
    try:
        response = self.mgmt_client.query(self.url, 'DELETE', self.query_parameters, self.header_parameters, None, self.status_code, 600, 30)
    except Exception as e:
        self.log('Error attempting to delete backup.')
        self.fail('Error deleting the azure backup: {0}'.format(str(e)))
    try:
        response = json.loads(response.body())
    except Exception:
        response = {'text': response.context['deserialized_data']}
    return response