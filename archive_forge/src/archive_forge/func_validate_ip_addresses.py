from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import netapp_ipaddress
def validate_ip_addresses(self):
    """
            Validate if the given IP address is a network address (i.e. it's host bits are set to 0)
            ONTAP doesn't validate if the host bits are set,
            and hence doesn't add a new address unless the IP is from a different network.
            So this validation allows the module to be idempotent.
            :return: None
        """
    for ip in self.parameters['allow_list']:
        netapp_ipaddress.validate_ip_address_is_network_address(ip, self.module)