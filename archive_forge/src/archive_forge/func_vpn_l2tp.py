from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def vpn_l2tp(data, fos):
    vdom = data['vdom']
    vpn_l2tp_data = data['vpn_l2tp']
    filtered_data = underscore_to_hyphen(filter_vpn_l2tp_data(vpn_l2tp_data))
    return fos.set('vpn', 'l2tp', data=filtered_data, vdom=vdom)