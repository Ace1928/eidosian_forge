from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def is_app_settings_changed(self):
    if self.app_settings:
        if self.app_settings_strDic:
            for key in self.app_settings.keys():
                if self.app_settings[key] != self.app_settings_strDic.get(key, None):
                    return True
        else:
            return True
    return False