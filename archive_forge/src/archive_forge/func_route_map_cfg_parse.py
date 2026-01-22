from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.route_maps.route_maps import Route_mapsArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import remove_empties_from_list
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic \
def route_map_cfg_parse(self, unparsed_route_map):
    """Parse the raw input configuration JSON representation for the route map specified
        by the "unparsed_route_map" input parameter. Parse the information to
        convert it to a dictionary matching the "argspec" for the "route_maps" resource
        module."""
    parsed_route_map_stmts = []
    if not unparsed_route_map.get('config'):
        return parsed_route_map_stmts
    route_map_name = unparsed_route_map.get('name')
    if not route_map_name:
        return parsed_route_map_stmts
    route_map_statements = unparsed_route_map.get('statements')
    if not route_map_statements:
        return parsed_route_map_stmts
    route_map_stmts_list = route_map_statements.get('statement')
    if not route_map_stmts_list:
        return parsed_route_map_stmts
    for route_map_stmt in route_map_stmts_list:
        parsed_route_map_stmt = {}
        parsed_seq_num = route_map_stmt.get('name')
        if not parsed_seq_num:
            continue
        parsed_route_map_stmt['map_name'] = route_map_name
        parsed_route_map_stmt['sequence_num'] = parsed_seq_num
        self.get_route_map_stmt_set_attr(route_map_stmt, parsed_route_map_stmt)
        self.get_route_map_stmt_match_attr(route_map_stmt, parsed_route_map_stmt)
        self.get_route_map_call_attr(route_map_stmt, parsed_route_map_stmt)
        parsed_route_map_stmts.append(parsed_route_map_stmt)
    return parsed_route_map_stmts