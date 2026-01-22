from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def to_restriction_obj(self, restriction_dict):
    return IpSecurityRestriction(name=restriction_dict['name'], description=restriction_dict['description'], action=restriction_dict['action'], priority=restriction_dict['priority'], ip_address=restriction_dict['ip_address'])