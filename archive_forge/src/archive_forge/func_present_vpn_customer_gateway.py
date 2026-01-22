from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def present_vpn_customer_gateway(self):
    vpn_customer_gateway = self.get_vpn_customer_gateway()
    required_params = ['cidrs', 'esp_policy', 'gateway', 'ike_policy', 'ipsec_psk']
    self.module.fail_on_missing_params(required_params=required_params)
    if not vpn_customer_gateway:
        vpn_customer_gateway = self._create_vpn_customer_gateway(vpn_customer_gateway)
    else:
        vpn_customer_gateway = self._update_vpn_customer_gateway(vpn_customer_gateway)
    return vpn_customer_gateway