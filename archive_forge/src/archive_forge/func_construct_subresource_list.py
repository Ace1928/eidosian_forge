from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, format_resource_id
from ansible.module_utils._text import to_native
def construct_subresource_list(self, raw):
    return [self.dns_models.SubResource(id=x) for x in raw] if raw else None