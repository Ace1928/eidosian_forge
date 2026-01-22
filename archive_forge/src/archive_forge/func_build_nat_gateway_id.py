from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, CIDR_PATTERN, azure_id_to_dict, format_resource_id
def build_nat_gateway_id(self, resource):
    """
        Common method to build a resource id from different inputs
        """
    if resource is None:
        return None
    if is_valid_resource_id(resource):
        return resource
    resource_dict = self.parse_resource_to_dict(resource)
    return format_resource_id(val=resource_dict['name'], subscription_id=resource_dict.get('subscription_id'), namespace='Microsoft.Network', types='natGateways', resource_group=resource_dict.get('resource_group'))