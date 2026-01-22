from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
def retention_policy_to_dict(self, policy):
    if policy:
        return dict(days=policy.get('days'), enabled=policy.get('enabled'))
    return None