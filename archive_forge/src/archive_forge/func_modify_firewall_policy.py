from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import netapp_ipaddress
def modify_firewall_policy(self, modify):
    """
        Modify a firewall Policy on a vserver
        :return: none
        """
    self.validate_ip_addresses()
    net_firewall_policy_obj = netapp_utils.zapi.NaElement('net-firewall-policy-modify')
    net_firewall_policy_obj.translate_struct(self.firewall_policy_attributes())
    net_firewall_policy_obj.add_child_elem(self.na_helper.get_value_for_list(from_zapi=False, zapi_parent='allow-list', zapi_child='ip-and-mask', data=modify['allow_list']))
    try:
        self.server.invoke_successfully(net_firewall_policy_obj, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error modifying Firewall Policy: %s' % to_native(error), exception=traceback.format_exc())