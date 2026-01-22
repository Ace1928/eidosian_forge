from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def system_affinity_interrupt(data, fos):
    vdom = data['vdom']
    state = data['state']
    system_affinity_interrupt_data = data['system_affinity_interrupt']
    filtered_data = underscore_to_hyphen(filter_system_affinity_interrupt_data(system_affinity_interrupt_data))
    if state == 'present' or state is True:
        return fos.set('system', 'affinity-interrupt', data=filtered_data, vdom=vdom)
    elif state == 'absent':
        return fos.delete('system', 'affinity-interrupt', mkey=filtered_data['id'], vdom=vdom)
    else:
        fos._module.fail_json(msg='state must be present or absent!')