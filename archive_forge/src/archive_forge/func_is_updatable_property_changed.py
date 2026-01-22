from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def is_updatable_property_changed(self, existing_webapp):
    for property_name in self.updatable_properties:
        if hasattr(self, property_name) and getattr(self, property_name) is not None and (getattr(self, property_name) != existing_webapp.get(property_name, None)):
            return True
    return False