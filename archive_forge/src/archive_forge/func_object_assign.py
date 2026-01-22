from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, format_resource_id
from ansible.module_utils._text import to_native
def object_assign(self, patch, origin):
    attribute_map = set(self.network_models.LoadBalancer._attribute_map.keys()) - set(self.network_models.LoadBalancer._validation.keys())
    for key in attribute_map:
        if not getattr(patch, key):
            setattr(patch, key, getattr(origin, key))
    return patch