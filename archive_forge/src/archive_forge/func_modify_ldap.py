from __future__ import absolute_import, division, print_function
import traceback
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
def modify_ldap(self, modify):
    """
        Modify LDAP
        :param modify: list of modify attributes
        """
    ldap_modify = netapp_utils.zapi.NaElement('ldap-config-modify')
    ldap_modify.add_new_child('client-config', self.parameters['name'])
    for attribute in modify:
        if attribute == 'skip_config_validation':
            ldap_modify.add_new_child('skip-config-validation', self.parameters[attribute])
    try:
        self.server.invoke_successfully(ldap_modify, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as errcatch:
        self.module.fail_json(msg='Error modifying LDAP %s: %s' % (self.parameters['name'], to_native(errcatch)), exception=traceback.format_exc())