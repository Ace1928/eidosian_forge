from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils._text import to_native
def format_to_dict(self, raw):
    return [self.account_obj_to_dict(item) for item in raw]