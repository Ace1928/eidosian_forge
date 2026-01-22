from __future__ import (absolute_import, division, print_function)
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.dict_transformations import dict_merge
from ansible_collections.community.general.plugins.module_utils.opennebula import flatten, render
def get_vm_info(client, vm):
    vm = client.vm.info(vm.ID)
    networks_info = []
    disk_size = []
    if 'DISK' in vm.TEMPLATE:
        if isinstance(vm.TEMPLATE['DISK'], list):
            for disk in vm.TEMPLATE['DISK']:
                disk_size.append(disk['SIZE'] + ' MB')
        else:
            disk_size.append(vm.TEMPLATE['DISK']['SIZE'] + ' MB')
    if 'NIC' in vm.TEMPLATE:
        if isinstance(vm.TEMPLATE['NIC'], list):
            for nic in vm.TEMPLATE['NIC']:
                networks_info.append({'ip': nic.get('IP', ''), 'mac': nic.get('MAC', ''), 'name': nic.get('NETWORK', ''), 'security_groups': nic.get('SECURITY_GROUPS', '')})
        else:
            networks_info.append({'ip': vm.TEMPLATE['NIC'].get('IP', ''), 'mac': vm.TEMPLATE['NIC'].get('MAC', ''), 'name': vm.TEMPLATE['NIC'].get('NETWORK', ''), 'security_groups': vm.TEMPLATE['NIC'].get('SECURITY_GROUPS', '')})
    import time
    current_time = time.localtime()
    vm_start_time = time.localtime(vm.STIME)
    vm_uptime = time.mktime(current_time) - time.mktime(vm_start_time)
    vm_uptime /= 60 * 60
    permissions_str = parse_vm_permissions(client, vm)
    vm_lcm_state = None
    if vm.STATE == VM_STATES.index('ACTIVE'):
        vm_lcm_state = LCM_STATES[vm.LCM_STATE]
    vm_labels, vm_attributes = get_vm_labels_and_attributes_dict(client, vm.ID)
    updateconf = parse_updateconf(vm.TEMPLATE)
    info = {'template_id': int(vm.TEMPLATE['TEMPLATE_ID']), 'vm_id': vm.ID, 'vm_name': vm.NAME, 'state': VM_STATES[vm.STATE], 'lcm_state': vm_lcm_state, 'owner_name': vm.UNAME, 'owner_id': vm.UID, 'networks': networks_info, 'disk_size': disk_size, 'memory': vm.TEMPLATE['MEMORY'] + ' MB', 'vcpu': vm.TEMPLATE['VCPU'], 'cpu': vm.TEMPLATE['CPU'], 'group_name': vm.GNAME, 'group_id': vm.GID, 'uptime_h': int(vm_uptime), 'attributes': vm_attributes, 'mode': permissions_str, 'labels': vm_labels, 'updateconf': updateconf}
    return info