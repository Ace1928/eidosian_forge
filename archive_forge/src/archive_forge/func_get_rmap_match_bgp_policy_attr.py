from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.route_maps.route_maps import Route_mapsArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import remove_empties_from_list
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic \
@staticmethod
def get_rmap_match_bgp_policy_attr(rmap_bgp_policy_match, parsed_rmap_match):
    """Parse the BGP policy "match" attribute portion of the raw input
        configuration JSON representation within the route map "statement"
        that is currently being parsed. The configuration section to be parsed
        is specified by the "rmap_bgp_match_cfg" input parameter. Parse the
        information to convert it to a dictionary matching the "argspec" for
        the "route_maps" resource module."""
    if rmap_bgp_policy_match.get('match-as-path-set') and rmap_bgp_policy_match['match-as-path-set'].get('config'):
        as_path = rmap_bgp_policy_match['match-as-path-set']['config'].get('as-path-set')
        if as_path:
            parsed_rmap_match['as_path'] = as_path
    rmap_bgp_match_cfg = rmap_bgp_policy_match.get('config')
    if rmap_bgp_match_cfg:
        match_metric = rmap_bgp_match_cfg.get('med-eq')
        if match_metric:
            parsed_rmap_match['metric'] = match_metric
        match_origin = rmap_bgp_match_cfg.get('origin-eq')
        if match_origin:
            if match_origin == 'IGP':
                parsed_rmap_match['origin'] = 'igp'
            elif match_origin == 'EGP':
                parsed_rmap_match['origin'] = 'egp'
            elif match_origin == 'INCOMPLETE':
                parsed_rmap_match['origin'] = 'incomplete'
        if rmap_bgp_match_cfg.get('local-pref-eq'):
            parsed_rmap_match['local_preference'] = rmap_bgp_match_cfg['local-pref-eq']
        if rmap_bgp_match_cfg.get('community-set'):
            parsed_rmap_match['community'] = rmap_bgp_match_cfg['community-set']
        if rmap_bgp_match_cfg.get('ext-community-set'):
            parsed_rmap_match['ext_comm'] = rmap_bgp_match_cfg['ext-community-set']
        if rmap_bgp_match_cfg.get('openconfig-bgp-policy-ext:next-hop-set'):
            parsed_rmap_match['ip'] = {}
            parsed_rmap_match['ip']['next_hop'] = rmap_bgp_match_cfg['openconfig-bgp-policy-ext:next-hop-set']
    if rmap_bgp_policy_match.get('openconfig-bgp-policy-ext:match-evpn-set'):
        bgp_policy_match_evpn_cfg = rmap_bgp_policy_match['openconfig-bgp-policy-ext:match-evpn-set'].get('config')
        if bgp_policy_match_evpn_cfg:
            parsed_rmap_match['evpn'] = {}
            if bgp_policy_match_evpn_cfg.get('vni-number'):
                parsed_rmap_match['evpn']['vni'] = bgp_policy_match_evpn_cfg.get('vni-number')
            if bgp_policy_match_evpn_cfg.get('default-type5-route'):
                parsed_rmap_match['evpn']['default_route'] = True
            evpn_route_type = bgp_policy_match_evpn_cfg.get('route-type')
            if evpn_route_type:
                if evpn_route_type == 'openconfig-bgp-policy-ext:MACIP':
                    parsed_rmap_match['evpn']['route_type'] = 'macip'
                elif evpn_route_type == 'openconfig-bgp-policy-ext:MULTICAST':
                    parsed_rmap_match['evpn']['route_type'] = 'multicast'
                elif evpn_route_type == 'openconfig-bgp-policy-ext:PREFIX':
                    parsed_rmap_match['evpn']['route_type'] = 'prefix'