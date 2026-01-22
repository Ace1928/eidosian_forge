from __future__ import absolute_import, division, print_function
import time
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, format_resource_id
def format_elastic_pool_id(self):
    parrent_id = format_resource_id(val=self.server_name, subscription_id=self.subscription_id, namespace='Microsoft.Sql', types='servers', resource_group=self.resource_group)
    self.parameters['elastic_pool_id'] = parrent_id + '/elasticPools/' + self.parameters['elastic_pool_id']