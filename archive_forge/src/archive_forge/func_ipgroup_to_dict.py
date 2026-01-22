from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import _load_params
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, normalize_location_name
def ipgroup_to_dict(self, ipgroup):
    result = ipgroup.as_dict()
    result['tags'] = ipgroup.tags
    return result