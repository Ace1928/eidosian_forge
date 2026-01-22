from __future__ import absolute_import, division, print_function
import time
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
def format_results(self, response):
    return {'id': response.get('id'), 'version': response.get('version'), 'state': response.get('state'), 'fully_qualified_domain_name': response.get('fully_qualified_domain_name')}