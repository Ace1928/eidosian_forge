from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.route_maps.route_maps import Route_mapsArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import remove_empties_from_list
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic \
@staticmethod
def get_rmap_set_extcommunity(set_bgp_policy, parsed_route_map_stmt_set):
    """Parse the "extcommunity" sub-section of the BGP policy "set"
        attribute portion of the raw input configuration JSON representation.
        The BGP policy "set" configuration section to be parsed is specified
        by the "set_bgp_policy" input parameter. Parse the information
        to convert it to a dictionary matching the "argspec" for the "route_maps"
        resource module."""
    set_extcommunity_top = set_bgp_policy.get('set-ext-community')
    if set_extcommunity_top and set_extcommunity_top.get('inline') and set_extcommunity_top['inline'].get('config') and set_extcommunity_top['inline']['config'].get('communities'):
        set_extcommunity_config_list = set_extcommunity_top['inline']['config']['communities']
        if set_extcommunity_config_list:
            parsed_route_map_stmt_set['extcommunity'] = {}
            parsed_rmap_stmt_set_extcomm = parsed_route_map_stmt_set['extcommunity']
            for set_extcommunity_config_item in set_extcommunity_config_list:
                if 'route-target:' in set_extcommunity_config_item:
                    rt_val = set_extcommunity_config_item.replace('route-target:', '')
                    if parsed_rmap_stmt_set_extcomm.get('rt'):
                        parsed_rmap_stmt_set_extcomm['rt'].append(rt_val)
                    else:
                        parsed_rmap_stmt_set_extcomm['rt'] = [rt_val]
                elif 'route-origin:' in set_extcommunity_config_item:
                    soo_val = set_extcommunity_config_item.replace('route-origin:', '')
                    if parsed_rmap_stmt_set_extcomm.get('soo'):
                        parsed_rmap_stmt_set_extcomm['soo'].append(soo_val)
                    else:
                        parsed_rmap_stmt_set_extcomm['soo'] = [soo_val]