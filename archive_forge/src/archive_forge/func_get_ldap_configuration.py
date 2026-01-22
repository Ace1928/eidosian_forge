from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
def get_ldap_configuration(self):
    """
            Return ldap configuration if found

            :return: Details about the ldap configuration. None if not found.
            :rtype: solidfire.models.GetLdapConfigurationResult
        """
    ldap_config = self.sfe.get_ldap_configuration()
    return ldap_config