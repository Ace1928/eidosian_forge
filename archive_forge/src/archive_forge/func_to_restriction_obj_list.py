from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def to_restriction_obj_list(self, restriction_dict_list):
    return [] if not restriction_dict_list else [self.to_restriction_obj(restriction) for restriction in restriction_dict_list]