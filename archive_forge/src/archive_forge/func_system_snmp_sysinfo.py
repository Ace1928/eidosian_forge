from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def system_snmp_sysinfo(data, fos):
    vdom = data['vdom']
    system_snmp_sysinfo_data = data['system_snmp_sysinfo']
    filtered_data = underscore_to_hyphen(filter_system_snmp_sysinfo_data(system_snmp_sysinfo_data))
    return fos.set('system.snmp', 'sysinfo', data=filtered_data, vdom=vdom)