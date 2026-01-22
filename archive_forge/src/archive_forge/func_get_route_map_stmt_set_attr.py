from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.route_maps.route_maps import Route_mapsArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import remove_empties_from_list
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic \
def get_route_map_stmt_set_attr(self, route_map_stmt, parsed_route_map_stmt):
    """Parse the "set" attribute portion of the raw input configuration JSON
        representation for the route map "statement" specified
        by the "route_map_stmt," input parameter. Parse the information to
        convert it to a dictionary matching the "argspec" for the "route_maps" resource
        module."""
    stmt_actions = route_map_stmt.get('actions')
    if not stmt_actions:
        return
    actions_config = stmt_actions.get('config')
    if not actions_config:
        return
    permit_deny_config = actions_config.get('policy-result')
    if not permit_deny_config:
        return
    if permit_deny_config == 'ACCEPT_ROUTE':
        parsed_route_map_stmt['action'] = 'permit'
    elif permit_deny_config == 'REJECT_ROUTE':
        parsed_route_map_stmt['action'] = 'deny'
    else:
        return
    parsed_route_map_stmt['set'] = {}
    parsed_route_map_stmt_set = parsed_route_map_stmt['set']
    set_metric_action = stmt_actions.get('metric-action')
    if set_metric_action:
        set_metric_action_cfg = set_metric_action.get('config')
        if set_metric_action_cfg:
            metric_action = set_metric_action_cfg.get('action')
            if metric_action:
                parsed_route_map_stmt_set['metric'] = {}
                if metric_action == 'openconfig-routing-policy:METRIC_SET_VALUE':
                    value = set_metric_action_cfg.get('metric')
                    if value:
                        parsed_route_map_stmt_set['metric']['value'] = value
                elif metric_action == 'openconfig-routing-policy:METRIC_SET_RTT':
                    parsed_route_map_stmt_set['metric']['rtt_action'] = 'set'
                elif metric_action == 'openconfig-routing-policy:METRIC_ADD_RTT':
                    parsed_route_map_stmt_set['metric']['rtt_action'] = 'add'
                elif metric_action == 'openconfig-routing-policy:METRIC_SUBTRACT_RTT':
                    parsed_route_map_stmt_set['metric']['rtt_action'] = 'subtract'
                if parsed_route_map_stmt_set['metric'] == {}:
                    parsed_route_map_stmt_set.pop('metric')
    set_bgp_policy = stmt_actions.get('openconfig-bgp-policy:bgp-actions')
    if set_bgp_policy:
        self.get_route_map_set_bgp_policy_attr(set_bgp_policy, parsed_route_map_stmt_set)