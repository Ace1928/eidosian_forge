from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def system_snmp_mib_view(data, fos):
    vdom = data['vdom']
    state = data['state']
    system_snmp_mib_view_data = data['system_snmp_mib_view']
    system_snmp_mib_view_data = flatten_multilists_attributes(system_snmp_mib_view_data)
    filtered_data = underscore_to_hyphen(filter_system_snmp_mib_view_data(system_snmp_mib_view_data))
    if state == 'present' or state is True:
        return fos.set('system.snmp', 'mib-view', data=filtered_data, vdom=vdom)
    elif state == 'absent':
        return fos.delete('system.snmp', 'mib-view', mkey=filtered_data['name'], vdom=vdom)
    else:
        fos._module.fail_json(msg='state must be present or absent!')