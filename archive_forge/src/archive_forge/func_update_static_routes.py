from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.static_routes.static_routes import Static_routesArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.bgp_utils import (
def update_static_routes(self, data):
    static_vrf_list = []
    for static_route in data:
        static_vrf_dict = {}
        static_route_list = static_route.get('static', [])
        vrf_name = static_route.get('vrf', None)
        static_list = []
        for static in static_route_list:
            static_dict = {}
            prefix = static.get('prefix', None)
            next_hops = static.get('next-hops', None)
            next_hop_list = next_hops.get('next-hop', [])
            next_hop_dict_list = []
            for next_hop in next_hop_list:
                next_hop_dict = {}
                index_dict = {}
                inf_ref = next_hop.get('interface-ref', {})
                inf_ref_cfg = inf_ref.get('config', {})
                interface = inf_ref_cfg.get('interface', None)
                config = next_hop.get('config', {})
                next_hop_attr = config.get('next-hop', None)
                metric = config.get('metric', None)
                nexthop_vrf = config.get('network-instance', None)
                blackhole = config.get('blackhole', None)
                track = config.get('track', None)
                tag = config.get('tag', None)
                if blackhole is not None:
                    index_dict['blackhole'] = blackhole
                if interface:
                    index_dict['interface'] = interface
                if nexthop_vrf:
                    index_dict['nexthop_vrf'] = nexthop_vrf
                if next_hop_attr:
                    index_dict['next_hop'] = next_hop_attr
                if index_dict:
                    next_hop_dict['index'] = index_dict
                if metric:
                    next_hop_dict['metric'] = metric
                if track:
                    next_hop_dict['track'] = track
                if tag:
                    next_hop_dict['tag'] = tag
                if next_hop_dict:
                    next_hop_dict_list.append(next_hop_dict)
            if prefix:
                static_dict['prefix'] = prefix
            if next_hop_dict_list:
                static_dict['next_hops'] = next_hop_dict_list
            if static_dict:
                static_list.append(static_dict)
        if static_list:
            static_vrf_dict['static_list'] = static_list
        if vrf_name:
            static_vrf_dict['vrf_name'] = vrf_name
        if static_vrf_dict:
            static_vrf_list.append(static_vrf_dict)
    return static_vrf_list