from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.community.general.plugins.module_utils.ldap import LdapGeneric, gen_specs, ldap_required_together
def passwd_set(self):
    if not self.passwd_check():
        return False
    try:
        self.connection.passwd_s(self.dn, None, self.passwd)
    except ldap.LDAPError as e:
        self.fail('Unable to set password', e)
    return True