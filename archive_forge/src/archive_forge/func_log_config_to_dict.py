from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
def log_config_to_dict(self, log_config):
    return dict(category=log_config.get('category'), category_group=log_config.get('category_group'), enabled=log_config.get('enabled'), retention_policy=self.retention_policy_to_dict(log_config.get('retention_policy')))