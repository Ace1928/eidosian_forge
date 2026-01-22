from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.comparison import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.comparison import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.comparison import (
def system_replacemsg_icap(data, fos, check_mode=False):
    vdom = data['vdom']
    state = data['state']
    system_replacemsg_icap_data = data['system_replacemsg_icap']
    filtered_data = underscore_to_hyphen(filter_system_replacemsg_icap_data(system_replacemsg_icap_data))
    if check_mode:
        diff = {'before': '', 'after': filtered_data}
        mkey = fos.get_mkey('system.replacemsg', 'icap', filtered_data, vdom=vdom)
        current_data = fos.get('system.replacemsg', 'icap', vdom=vdom, mkey=mkey)
        is_existed = current_data and current_data.get('http_status') == 200 and isinstance(current_data.get('results'), list) and (len(current_data['results']) > 0)
        if state == 'present' or state is True:
            if mkey is None:
                return (False, True, filtered_data, diff)
            if is_existed:
                is_same = is_same_comparison(serialize(current_data['results'][0]), serialize(filtered_data))
                current_values = find_current_values(current_data['results'][0], filtered_data)
                return (False, not is_same, filtered_data, {'before': current_values, 'after': filtered_data})
            return (False, True, filtered_data, diff)
        if state == 'absent':
            if mkey is None:
                return (False, False, filtered_data, {'before': current_data['results'][0], 'after': ''})
            if is_existed:
                return (False, True, filtered_data, {'before': current_data['results'][0], 'after': ''})
            return (False, False, filtered_data, {})
        return (True, False, {'reason: ': 'Must provide state parameter'}, {})
    if state == 'present' or state is True:
        return fos.set('system.replacemsg', 'icap', data=filtered_data, vdom=vdom)
    elif state == 'absent':
        return fos.delete('system.replacemsg', 'icap', mkey=filtered_data['msg-type'], vdom=vdom)
    else:
        fos._module.fail_json(msg='state must be present or absent!')