from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.vxlans.vxlans import VxlansArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_all_vxlans(self):
    vxlans = []
    vxlan_tunnels = []
    vxlan_vlan_map = []
    vxlans_tunnels_vlan_map = self.get_all_vxlans_tunnels_vlan_map()
    vxlans_evpn_nvo_list = self.get_all_vxlans_evpn_nvo_list()
    if vxlans_tunnels_vlan_map.get('VXLAN_TUNNEL'):
        if vxlans_tunnels_vlan_map['VXLAN_TUNNEL'].get('VXLAN_TUNNEL_LIST'):
            vxlan_tunnels.extend(vxlans_tunnels_vlan_map['VXLAN_TUNNEL']['VXLAN_TUNNEL_LIST'])
    if vxlans_tunnels_vlan_map.get('VXLAN_TUNNEL_MAP'):
        if vxlans_tunnels_vlan_map['VXLAN_TUNNEL_MAP'].get('VXLAN_TUNNEL_MAP_LIST'):
            vxlan_vlan_map.extend(vxlans_tunnels_vlan_map['VXLAN_TUNNEL_MAP']['VXLAN_TUNNEL_MAP_LIST'])
    self.fill_tunnel_source_ip(vxlans, vxlan_tunnels, vxlans_evpn_nvo_list)
    self.fill_vlan_map(vxlans, vxlan_vlan_map)
    vxlan_vrf_list = self.get_all_vxlans_vrf_list()
    self.fill_vrf_map(vxlans, vxlan_vrf_list)
    return vxlans