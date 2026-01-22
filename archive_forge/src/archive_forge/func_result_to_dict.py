from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBase
def result_to_dict(self, object):
    if object.object_type == 'Group':
        return self.group_to_dict(object)
    elif object.object_type == 'User':
        return self.user_to_dict(object)
    elif object.object_type == 'Application':
        return self.application_to_dict(object)
    elif object.object_type == 'ServicePrincipal':
        return self.serviceprincipal_to_dict(object)
    else:
        return object.object_type