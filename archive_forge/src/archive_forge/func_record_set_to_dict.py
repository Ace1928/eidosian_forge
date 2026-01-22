from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
def record_set_to_dict(self, record_set):
    return dict(fqdn=record_set.get('fqdn'), ip_addresses=record_set.get('ip_addresses'), provisioning_state=record_set.get('provisioning_state'), record_set_name=record_set.get('record_set_name'), record_type=record_set.get('record_type'), ttl=record_set.get('ttl'))