from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def to_restriction_dict(self, restriction_obj):
    return dict(name=restriction_obj.name, description=restriction_obj.description, action=restriction_obj.action, priority=restriction_obj.priority, ip_address=restriction_obj.ip_address)