from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def is_deployment_source_changed(self, existing_webapp):
    if self.deployment_source:
        if self.deployment_source.get('url') and self.deployment_source['url'] != existing_webapp.get('site_source_control')['url']:
            return True
        if self.deployment_source.get('branch') and self.deployment_source['branch'] != existing_webapp.get('site_source_control')['branch']:
            return True
    return False