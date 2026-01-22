from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import netapp_ipaddress
def modify_firewall_config(self, modify):
    """
        Modify the configuration of a firewall on node
        :return: None
        """
    net_firewall_config_obj = netapp_utils.zapi.NaElement('net-firewall-config-modify')
    net_firewall_config_obj.add_new_child('node-name', self.parameters['node'])
    if modify.get('enable'):
        net_firewall_config_obj.add_new_child('is-enabled', self.change_status_to_bool(self.parameters['enable']))
    if modify.get('logging'):
        net_firewall_config_obj.add_new_child('is-logging', self.change_status_to_bool(self.parameters['logging']))
    try:
        self.server.invoke_successfully(net_firewall_config_obj, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error modifying Firewall Config: %s' % to_native(error), exception=traceback.format_exc())