from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import _load_params
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, normalize_location_name
def ip_addresses_changed(self, input_records, ip_group_records):
    input_set = set(input_records)
    ip_group_set = set(ip_group_records)
    changed = input_set != ip_group_set
    return changed