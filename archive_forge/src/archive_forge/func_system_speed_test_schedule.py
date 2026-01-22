from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def system_speed_test_schedule(data, fos):
    vdom = data['vdom']
    state = data['state']
    system_speed_test_schedule_data = data['system_speed_test_schedule']
    filtered_data = underscore_to_hyphen(filter_system_speed_test_schedule_data(system_speed_test_schedule_data))
    if state == 'present' or state is True:
        return fos.set('system', 'speed-test-schedule', data=filtered_data, vdom=vdom)
    elif state == 'absent':
        return fos.delete('system', 'speed-test-schedule', mkey=filtered_data['interface'], vdom=vdom)
    else:
        fos._module.fail_json(msg='state must be present or absent!')