from __future__ import absolute_import, division, print_function
import traceback
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_krbrealm(self):
    """supported
        Create Kerberos Realm configuration
        """
    if self.use_rest:
        return self.create_krbrealm_rest()
    options = {'realm': self.parameters['realm']}
    for attribute in self.simple_attributes:
        if self.parameters.get(attribute) is not None:
            options[str(attribute).replace('_', '-')] = self.parameters[attribute]
    if self.parameters.get('kdc_port'):
        options['kdc-port'] = str(self.parameters['kdc_port'])
    if self.parameters.get('pw_server_ip') is not None:
        options['password-server-ip'] = self.parameters['pw_server_ip']
    if self.parameters.get('pw_server_port') is not None:
        options['password-server-port'] = self.parameters['pw_server_port']
    if self.parameters.get('ad_server_ip') is not None:
        options['ad-server-ip'] = self.parameters['ad_server_ip']
    if self.parameters.get('ad_server_name') is not None:
        options['ad-server-name'] = self.parameters['ad_server_name']
    krbrealm_create = netapp_utils.zapi.NaElement.create_node_with_children('kerberos-realm-create', **options)
    try:
        self.server.invoke_successfully(krbrealm_create, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as errcatch:
        self.module.fail_json(msg='Error creating Kerberos Realm configuration %s: %s' % (self.parameters['realm'], to_native(errcatch)), exception=traceback.format_exc())