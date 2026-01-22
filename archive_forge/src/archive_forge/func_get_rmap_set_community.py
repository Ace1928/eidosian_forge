from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.route_maps.route_maps import Route_mapsArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import remove_empties_from_list
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic \
@staticmethod
def get_rmap_set_community(set_bgp_policy, parsed_route_map_stmt_set):
    """Parse the "community" sub-section of the BGP policy "set" attribute
        portion of the raw input configuration JSON representation.
        The BGP policy "set" configuration section to be parsed is specified
        by the "set_bgp_policy" input parameter. Parse the information
        to convert it to a dictionary matching the "argspec" for the "route_maps"
        resource module."""
    set_community_top = set_bgp_policy.get('set-community')
    if set_community_top and set_community_top.get('inline') and set_community_top['inline'].get('config') and set_community_top['inline']['config'].get('communities'):
        set_community_config_list = set_community_top['inline']['config']['communities']
        parsed_route_map_stmt_set['community'] = {}
        parsed_rmap_stmt_set_comm = parsed_route_map_stmt_set['community']
        for set_community_config_item in set_community_config_list:
            if set_community_config_item.split(':')[0] in ('openconfig-bgp-types', 'openconfig-routing-policy-ext'):
                set_community_attr = set_community_config_item.split(':')[1]
                if not parsed_rmap_stmt_set_comm.get('community_attributes'):
                    parsed_rmap_stmt_set_comm['community_attributes'] = []
                    parsed_comm_attr_list = parsed_rmap_stmt_set_comm['community_attributes']
                comm_attr_rest_to_argspec = {'NO_EXPORT_SUBCONFED': 'local_as', 'NO_ADVERTISE': 'no_advertise', 'NO_EXPORT': 'no_export', 'NOPEER': 'no_peer', 'NONE': 'none', 'ADDITIVE': 'additive'}
                if set_community_attr in comm_attr_rest_to_argspec:
                    parsed_comm_attr_list.append(comm_attr_rest_to_argspec[set_community_attr])
            else:
                if not parsed_rmap_stmt_set_comm.get('community_number'):
                    parsed_rmap_stmt_set_comm['community_number'] = []
                    parsed_comm_num_list = parsed_rmap_stmt_set_comm['community_number']
                set_community_num_val_match = re.match('\\d+:\\d+$', set_community_config_item)
                if set_community_num_val_match:
                    parsed_comm_num_list.append(set_community_config_item)