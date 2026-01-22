from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import (AzureRMModuleBase,
from ansible.module_utils._text import to_native
def backend_addr_pool_id(self, val):
    if isinstance(val, dict):
        lb = val.get('load_balancer', None)
        name = val.get('name', None)
        if lb and name:
            return resource_id(subscription=self.subscription_id, resource_group=self.resource_group, namespace='Microsoft.Network', type='loadBalancers', name=lb, child_type_1='backendAddressPools', child_name_1=name)
    return val