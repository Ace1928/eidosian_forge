from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
def diagnostic_setting_to_dict(self, diagnostic_setting):
    setting_dict = diagnostic_setting if isinstance(diagnostic_setting, dict) else diagnostic_setting.as_dict()
    result = dict(id=setting_dict.get('id'), name=setting_dict.get('name'), event_hub=self.event_hub_dict(setting_dict), storage_account=self.storage_dict(setting_dict.get('storage_account_id')), log_analytics=self.log_analytics_dict(setting_dict.get('workspace_id')), logs=[self.log_config_to_dict(log) for log in setting_dict.get('logs', [])], metrics=[self.metric_config_to_dict(metric) for metric in setting_dict.get('metrics', [])])
    return self.remove_disabled_config(result)