from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import netapp_ipaddress
def destroy_firewall_policy(self):
    """
        Destroy a Firewall Policy from a vserver
        :return: None
        """
    net_firewall_policy_obj = netapp_utils.zapi.NaElement('net-firewall-policy-destroy')
    net_firewall_policy_obj.translate_struct(self.firewall_policy_attributes())
    try:
        self.server.invoke_successfully(net_firewall_policy_obj, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error destroying Firewall Policy: %s' % to_native(error), exception=traceback.format_exc())