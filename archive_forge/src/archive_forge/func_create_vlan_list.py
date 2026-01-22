from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def create_vlan_list(self):
    vlan_id_list = []
    for vlan_id_splitted in self.module.params['vlan_id'].split(','):
        vlans = vlan_id_splitted.split('-')
        if len(vlans) > 2:
            self.module.fail_json(msg='Invalid VLAN range %s.' % vlan_id_splitted)
        if len(vlans) == 2:
            vlan_id_start = vlans[0].strip()
            vlan_id_end = vlans[1].strip()
            if not vlan_id_start.isdigit():
                self.module.fail_json(msg='Invalid VLAN %s.' % vlan_id_start)
            if not vlan_id_end.isdigit():
                self.module.fail_json(msg='Invalid VLAN %s.' % vlan_id_end)
            vlan_id_start = int(vlan_id_start)
            vlan_id_end = int(vlan_id_end)
            if vlan_id_start not in range(0, 4095) or vlan_id_end not in range(0, 4095):
                self.module.fail_json(msg='vlan_id range %s specified is incorrect. The valid vlan_id range is from 0 to 4094.' % vlan_id_splitted)
            vlan_id_list.append((vlan_id_start, vlan_id_end))
        else:
            vlan_id = vlans[0].strip()
            if not vlan_id.isdigit():
                self.module.fail_json(msg='Invalid VLAN %s.' % vlan_id)
            vlan_id = int(vlan_id)
            vlan_id_list.append((vlan_id, vlan_id))
    vlan_id_list.sort()
    return vlan_id_list