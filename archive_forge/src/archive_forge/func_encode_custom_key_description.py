from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
def encode_custom_key_description(self, key_description):
    return key_description.encode('utf-16')