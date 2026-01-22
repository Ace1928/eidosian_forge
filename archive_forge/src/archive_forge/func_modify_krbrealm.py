from __future__ import absolute_import, division, print_function
import traceback
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_krbrealm(self, modify):
    """
        Modify Kerberos Realm
        :param modify: list of modify attributes
        """
    if self.use_rest:
        return self.modify_krbrealm_rest(modify)
    krbrealm_modify = netapp_utils.zapi.NaElement('kerberos-realm-modify')
    krbrealm_modify.add_new_child('realm', self.parameters['realm'])
    for attribute in modify:
        if attribute in self.simple_attributes:
            krbrealm_modify.add_new_child(str(attribute).replace('_', '-'), self.parameters[attribute])
        if attribute == 'kdc_port':
            krbrealm_modify.add_new_child('kdc-port', str(self.parameters['kdc_port']))
        if attribute == 'pw_server_ip':
            krbrealm_modify.add_new_child('password-server-ip', self.parameters['pw_server_ip'])
        if attribute == 'pw_server_port':
            krbrealm_modify.add_new_child('password-server-port', self.parameters['pw_server_port'])
        if attribute == 'ad_server_ip':
            krbrealm_modify.add_new_child('ad-server-ip', self.parameters['ad_server_ip'])
        if attribute == 'ad_server_name':
            krbrealm_modify.add_new_child('ad-server-name', self.parameters['ad_server_name'])
        if attribute == 'comment':
            krbrealm_modify.add_new_child('comment', self.parameters['comment'])
    try:
        self.server.invoke_successfully(krbrealm_modify, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as errcatch:
        self.module.fail_json(msg='Error modifying Kerberos Realm %s: %s' % (self.parameters['realm'], to_native(errcatch)), exception=traceback.format_exc())