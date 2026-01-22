from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.route_maps.route_maps import Route_mapsArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import remove_empties_from_list
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic \
@staticmethod
def get_route_map_call_attr(route_map_stmt, parsed_route_map_stmt):
    """Parse the "call" attribute portion of the raw input configuration JSON
        representation for the route map "statement" specified
        by the "route_map_stmt," input parameter. Parse the information to
        convert it to a dictionary matching the "argspec" for the "route_maps" resource
        module."""
    stmt_conditions = route_map_stmt.get('conditions')
    if not stmt_conditions:
        return
    conditions_config = stmt_conditions.get('config')
    if not conditions_config:
        return
    call_str = conditions_config.get('call-policy')
    if not call_str:
        return
    parsed_route_map_stmt['call'] = call_str