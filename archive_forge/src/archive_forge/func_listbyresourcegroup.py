from __future__ import absolute_import, division, print_function
import json
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_rest import GenericRestClient
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
def listbyresourcegroup(self):
    response = None
    results = {}
    self.url = '/subscriptions' + '/{{ subscription_id }}' + '/resourceGroups' + '/{{ resource_group }}' + '/providers' + '/Microsoft.ApiManagement' + '/service'
    self.url = self.url.replace('{{ subscription_id }}', self.subscription_id)
    self.url = self.url.replace('{{ resource_group }}', self.resource_group)
    try:
        response = self.mgmt_client.query(self.url, 'GET', self.query_parameters, self.header_parameters, None, self.status_code, 600, 30)
        results = json.loads(response.body())
    except Exception as e:
        self.log('Could not get info for @(Model.ModuleOperationNameUpper).')
    return [self.format_item(x) for x in results['value']] if results['value'] else []