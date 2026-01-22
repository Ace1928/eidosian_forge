from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def get_updated_config(self, site_config):
    site_config.ip_security_restrictions = [] if not self.ip_security_restrictions else self.to_restriction_obj_list(self.ip_security_restrictions)
    site_config.scm_ip_security_restrictions = [] if not self.scm_ip_security_restrictions else self.to_restriction_obj_list(self.scm_ip_security_restrictions)
    site_config.scm_ip_security_restrictions_use_main = self.scm_ip_security_restrictions_use_main
    return site_config