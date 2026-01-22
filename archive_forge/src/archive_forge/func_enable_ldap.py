from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
def enable_ldap(self):
    """
        Enable LDAP
        :return: nothing
        """
    try:
        self.sfe.enable_ldap_authentication(self.serverURIs, auth_type=self.authType, group_search_base_dn=self.groupSearchBaseDn, group_search_type=self.groupSearchType, group_search_custom_filter=self.groupSearchCustomFilter, search_bind_dn=self.searchBindDN, search_bind_password=self.searchBindPassword, user_search_base_dn=self.userSearchBaseDN, user_search_filter=self.userSearchFilter, user_dntemplate=self.userDNTemplate)
    except solidfire.common.ApiServerError as error:
        self.module.fail_json(msg='Error enabling LDAP: %s' % to_native(error), exception=traceback.format_exc())