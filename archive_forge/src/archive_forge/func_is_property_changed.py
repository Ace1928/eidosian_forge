from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def is_property_changed(profile, property, ignore_case=False):
    base = response[profile].get(property)
    new = getattr(self, profile).get(property)
    if ignore_case:
        return base.lower() != new.lower()
    else:
        return base != new