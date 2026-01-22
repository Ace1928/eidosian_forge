from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, format_resource_id
from ansible.module_utils._text import to_native
def parse_vn_id(self, vn):
    vn_dict = self.parse_resource_to_dict(vn) if not isinstance(vn, dict) else vn
    return format_resource_id(val=vn_dict['name'], subscription_id=vn_dict.get('subscription') or self.subscription_id, namespace='Microsoft.Network', types='virtualNetworks', resource_group=vn_dict.get('resource_group') or self.resource_group)