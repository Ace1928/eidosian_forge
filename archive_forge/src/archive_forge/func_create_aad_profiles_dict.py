from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def create_aad_profiles_dict(aad):
    return aad.as_dict() if aad else dict()