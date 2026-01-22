from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def vpn_ike_gateway(data, fos):
    vdom = data['vdom']
    vpn_ike_gateway_data = data['vpn_ike_gateway']
    filtered_data = underscore_to_hyphen(filter_vpn_ike_gateway_data(vpn_ike_gateway_data))
    converted_data = valid_attr_to_invalid_attrs(filtered_data)
    return fos.set('vpn.ike', 'gateway', data=converted_data, vdom=vdom)