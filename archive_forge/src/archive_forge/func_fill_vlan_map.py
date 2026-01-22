from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.vxlans.vxlans import VxlansArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def fill_vlan_map(self, vxlans, vxlan_vlan_map):
    for each_vlan_map in vxlan_vlan_map:
        name = each_vlan_map['name']
        matched_vtep = next((each_vxlan for each_vxlan in vxlans if each_vxlan['name'] == name), None)
        if matched_vtep:
            vni = int(each_vlan_map['vni'])
            vlan = int(each_vlan_map['vlan'][4:])
            vlan_map = matched_vtep.get('vlan_map')
            if vlan_map:
                vlan_map.append(dict({'vni': vni, 'vlan': vlan}))
            else:
                matched_vtep['vlan_map'] = [dict({'vni': vni, 'vlan': vlan})]