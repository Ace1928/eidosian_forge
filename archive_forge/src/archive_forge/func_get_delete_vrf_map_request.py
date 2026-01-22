from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible.module_utils.connection import ConnectionError
def get_delete_vrf_map_request(self, conf, matched, name, del_vrf_map_list):
    requests = []
    for each_vrf_map in del_vrf_map_list:
        vrf = each_vrf_map.get('vrf')
        vni = each_vrf_map.get('vni')
        is_change_needed = False
        if matched:
            matched_vrf_map_list = matched.get('vrf_map', None)
            if matched_vrf_map_list:
                matched_vrf_map = next((e_vrf_map for e_vrf_map in matched_vrf_map_list if e_vrf_map['vni'] == vni), None)
                if matched_vrf_map:
                    if matched_vrf_map['vrf'] == vrf:
                        is_change_needed = True
        if is_change_needed:
            url = 'data/sonic-vrf:sonic-vrf/VRF/VRF_LIST={vrf}/vni'.format(vrf=vrf)
            request = {'path': url, 'method': DELETE}
            requests.append(request)
    return requests