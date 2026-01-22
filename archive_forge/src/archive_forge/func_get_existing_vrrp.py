from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def get_existing_vrrp(interface, group, module, name):
    command = 'show vrrp detail interface {0}'.format(interface)
    body = execute_show_command(command, module)
    vrrp = {}
    vrrp_key = {'sh_group_id': 'group', 'sh_vip_addr': 'vip', 'sh_priority': 'priority', 'sh_group_preempt': 'preempt', 'sh_auth_text': 'authentication', 'sh_adv_interval': 'interval'}
    try:
        vrrp_table = body['TABLE_vrrp_group']
    except (AttributeError, IndexError, TypeError):
        return {}
    if isinstance(vrrp_table, dict):
        vrrp_table = [vrrp_table]
    for each_vrrp in vrrp_table:
        vrrp_row = each_vrrp['ROW_vrrp_group']
        parsed_vrrp = apply_key_map(vrrp_key, vrrp_row)
        if parsed_vrrp['preempt'] == 'Disable':
            parsed_vrrp['preempt'] = False
        elif parsed_vrrp['preempt'] == 'Enable':
            parsed_vrrp['preempt'] = True
        if parsed_vrrp['group'] == group:
            parsed_vrrp['admin_state'] = get_vrr_status(group, module, name)
            return parsed_vrrp
    return vrrp