from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.bgp_neighbors_af.bgp_neighbors_af import Bgp_neighbors_afArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.bgp_utils import (
def normalize_neighbors_af_data(self, neighbors):
    norm_neighbors = []
    for nei_data in neighbors:
        norm_neighbor = {}
        neighbor = nei_data.get('neighbor-address', None)
        if not neighbor:
            continue
        norm_neighbor['neighbor'] = neighbor
        norm_neighbor['address_family'] = []
        nei_afs = nei_data.get('afi-safis', None)
        if not nei_afs:
            if norm_neighbor:
                norm_neighbors.append(norm_neighbor)
            continue
        nei_afs = nei_afs.get('afi-safi', None)
        if not nei_afs:
            if norm_neighbor:
                norm_neighbors.append(norm_neighbor)
            continue
        norm_neighbor_afs = []
        for nei_af in nei_afs:
            norm_nei_af = get_from_params_map(self.neighbor_af_params_map, nei_af)
            if norm_nei_af:
                if 'activate' not in norm_nei_af:
                    norm_nei_af['activate'] = False
                if 'route_server_client' not in norm_nei_af:
                    norm_nei_af['route_server_client'] = False
                norm_nei_af['route_map'] = []
                self.fill_route_map(norm_nei_af)
                allowas_in = {}
                allowas_in_origin = norm_nei_af.get('allowas_in_origin', None)
                if allowas_in_origin is not None:
                    allowas_in['origin'] = allowas_in_origin
                    norm_nei_af.pop('allowas_in_origin')
                allowas_in_value = norm_nei_af.get('allowas_in_value', None)
                if allowas_in_value is not None:
                    allowas_in['value'] = allowas_in_value
                    norm_nei_af.pop('allowas_in_value')
                if allowas_in:
                    norm_nei_af['allowas_in'] = allowas_in
                ipv4_unicast = norm_nei_af.get('ipv4_unicast', None)
                ipv6_unicast = norm_nei_af.get('ipv6_unicast', None)
                if ipv4_unicast:
                    if 'config' in ipv4_unicast:
                        ip_afi = update_bgp_nbr_pg_ip_afi_dict(ipv4_unicast['config'])
                        if ip_afi:
                            norm_nei_af['ip_afi'] = ip_afi
                    if 'prefix-limit' in ipv4_unicast and 'config' in ipv4_unicast['prefix-limit']:
                        prefix_limit = update_bgp_nbr_pg_prefix_limit_dict(ipv4_unicast['prefix-limit']['config'])
                        if prefix_limit:
                            norm_nei_af['prefix_limit'] = prefix_limit
                    norm_nei_af.pop('ipv4_unicast')
                elif ipv6_unicast:
                    if 'config' in ipv6_unicast:
                        ip_afi = update_bgp_nbr_pg_ip_afi_dict(ipv6_unicast['config'])
                        if ip_afi:
                            norm_nei_af['ip_afi'] = ip_afi
                    if 'prefix-limit' in ipv6_unicast and 'config' in ipv6_unicast['prefix-limit']:
                        prefix_limit = update_bgp_nbr_pg_prefix_limit_dict(ipv6_unicast['prefix-limit']['config'])
                        if prefix_limit:
                            norm_nei_af['prefix_limit'] = prefix_limit
                    norm_nei_af.pop('ipv6_unicast')
                norm_neighbor_afs.append(norm_nei_af)
        if norm_neighbor_afs:
            norm_neighbor['address_family'] = norm_neighbor_afs
        if norm_neighbor:
            norm_neighbors.append(norm_neighbor)
    return norm_neighbors