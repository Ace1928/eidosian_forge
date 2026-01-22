from __future__ import absolute_import, division, print_function
import traceback
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_krbrealm(self):
    """
        Checks if Kerberos Realm config exists.

        :return:
            kerberos realm object if found
            None if not found
        :rtype: object/None
        """
    if self.use_rest:
        return self.get_krbrealm_rest()
    krbrealm_info = netapp_utils.zapi.NaElement('kerberos-realm-get-iter')
    query_details = netapp_utils.zapi.NaElement.create_node_with_children('kerberos-realm', **{'realm': self.parameters['realm'], 'vserver-name': self.parameters['vserver']})
    query = netapp_utils.zapi.NaElement('query')
    query.add_child_elem(query_details)
    krbrealm_info.add_child_elem(query)
    try:
        result = self.server.invoke_successfully(krbrealm_info, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error fetching kerberos realm %s: %s' % (self.parameters['realm'], to_native(error)))
    krbrealm_details = None
    if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) >= 1:
        attributes_list = result.get_child_by_name('attributes-list')
        config_info = attributes_list.get_child_by_name('kerberos-realm')
        krbrealm_details = {'admin_server_ip': config_info.get_child_content('admin-server-ip'), 'admin_server_port': config_info.get_child_content('admin-server-port'), 'clock_skew': config_info.get_child_content('clock-skew'), 'kdc_ip': config_info.get_child_content('kdc-ip'), 'kdc_port': int(config_info.get_child_content('kdc-port')), 'kdc_vendor': config_info.get_child_content('kdc-vendor'), 'pw_server_ip': config_info.get_child_content('password-server-ip'), 'pw_server_port': config_info.get_child_content('password-server-port'), 'realm': config_info.get_child_content('realm'), 'vserver': config_info.get_child_content('vserver-name'), 'ad_server_ip': config_info.get_child_content('ad-server-ip'), 'ad_server_name': config_info.get_child_content('ad-server-name'), 'comment': config_info.get_child_content('comment')}
    return krbrealm_details