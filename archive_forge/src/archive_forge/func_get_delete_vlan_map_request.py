from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible.module_utils.connection import ConnectionError
def get_delete_vlan_map_request(self, conf, matched, name, del_vlan_map_list):
    requests = []
    for each_vlan_map in del_vlan_map_list:
        vlan = each_vlan_map.get('vlan')
        vni = each_vlan_map.get('vni')
        is_change_needed = False
        if matched:
            matched_vlan_map_list = matched.get('vlan_map', None)
            if matched_vlan_map_list:
                matched_vlan_map = next((e_vlan_map for e_vlan_map in matched_vlan_map_list if e_vlan_map['vni'] == vni), None)
                if matched_vlan_map:
                    if matched_vlan_map['vlan'] == vlan:
                        is_change_needed = True
        if is_change_needed:
            map_name = 'map_{0}_Vlan{1}'.format(vni, vlan)
            url = 'data/sonic-vxlan:sonic-vxlan/VXLAN_TUNNEL_MAP/VXLAN_TUNNEL_MAP_LIST={name},{map_name}'.format(name=name, map_name=map_name)
            request = {'path': url, 'method': DELETE}
            requests.append(request)
    return requests