from __future__ import absolute_import, division, print_function
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_all_bgp_neighbors(module):
    vrf_list = get_all_vrfs(module)
    'Get all BGP neighbor configurations available in chassis'
    all_bgp_neighbors = []
    for vrf_name in vrf_list:
        neighbors_cfg = {}
        bgp_as = get_bgp_as(module, vrf_name)
        if bgp_as:
            neighbors_cfg['bgp_as'] = bgp_as
            neighbors_cfg['vrf_name'] = vrf_name
        else:
            continue
        neighbors = get_bgp_neighbors(module, vrf_name)
        if neighbors:
            neighbors_cfg['neighbors'] = neighbors
        if neighbors_cfg:
            all_bgp_neighbors.append(neighbors_cfg)
    return all_bgp_neighbors