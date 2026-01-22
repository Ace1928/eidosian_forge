from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def switch_controller_snmp_sysinfo(data, fos):
    vdom = data['vdom']
    switch_controller_snmp_sysinfo_data = data['switch_controller_snmp_sysinfo']
    filtered_data = underscore_to_hyphen(filter_switch_controller_snmp_sysinfo_data(switch_controller_snmp_sysinfo_data))
    return fos.set('switch-controller', 'snmp-sysinfo', data=filtered_data, vdom=vdom)