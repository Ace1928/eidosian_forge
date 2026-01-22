from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
def get_route_map_modify_set_attr(self, command, route_map_statement, have):
    """In the dict specified by the input route_map_statement paramenter,
        provide REST API definitions of all "set" attributes contained in the
        user input command dict specified by the "command" input parameter
        to this function."""
    cmd_set_top = command.get('set')
    if not cmd_set_top:
        return
    cfg_set_top = {}
    conf_map_name = command.get('map_name')
    conf_seq_num = command.get('sequence_num')
    cmd_rmap_have = self.get_matching_map(conf_map_name, conf_seq_num, have)
    if cmd_rmap_have:
        cfg_set_top = cmd_rmap_have.get('set')
    route_map_actions = route_map_statement['actions']
    route_map_actions['openconfig-bgp-policy:bgp-actions'] = {}
    route_map_bgp_actions = route_map_actions['openconfig-bgp-policy:bgp-actions'] = {}
    if cmd_set_top.get('as_path_prepend'):
        route_map_bgp_actions['set-as-path-prepend'] = {'config': {'openconfig-routing-policy-ext:asn-list': cmd_set_top['as_path_prepend']}}
    if cmd_set_top.get('comm_list_delete'):
        route_map_bgp_actions['set-community-delete'] = {'config': {'community-set-delete': cmd_set_top['comm_list_delete']}}
    if cmd_set_top.get('community'):
        route_map_bgp_actions['set-community'] = {'config': {'method': 'INLINE', 'options': 'ADD'}, 'inline': {'config': {'communities': []}}}
        rmap_set_communities_cfg = route_map_bgp_actions['set-community']['inline']['config']['communities']
        if cmd_set_top['community'].get('community_number'):
            if cfg_set_top:
                if cfg_set_top.get('community') and cfg_set_top['community'].get('community_attributes') and ('none' in cfg_set_top['community']['community_attributes']):
                    self._module.fail_json(msg='\nPlaybook aborted: The route map "set" community "none" attribute is configured.\n\nPlease remove the conflicting configuration to configure other community "set" attributes.\n')
            comm_num_list = cmd_set_top['community']['community_number']
            for comm_num in comm_num_list:
                rmap_set_communities_cfg.append(comm_num)
        if cmd_set_top['community'].get('community_attributes'):
            comm_attr_list = []
            comm_attr_list = cmd_set_top['community']['community_attributes']
            if 'none' in comm_attr_list:
                if len(comm_attr_list) > 1 or rmap_set_communities_cfg:
                    self._module.fail_json(msg='\nPlaybook aborted: The route map "set" community "none"attribute cannot be configured when other "set" community attributes are requested or configured.\n\nPlease revise the playbook to configure the "none"attribute.\n')
                if cfg_set_top:
                    if cfg_set_top.get('community') and (cfg_set_top['community'].get('community_number') or (cfg_set_top['community'].get('community_attributes') and 'none' not in cfg_set_top['community']['community_attributes'])):
                        self._module.fail_json(msg='\nPlaybook aborted: The route map "set" community "none"  attribute cannot be configured when other"set" community attributes are requested or configured.\n\nPlease remove the conflicting configuration to configure the "none" attribue.\n')
                rmap_set_communities_cfg.append('openconfig-bgp-types:NONE')
            else:
                if cfg_set_top:
                    if cfg_set_top.get('community') and cfg_set_top['community'].get('community_attributes') and ('none' in cfg_set_top['community']['community_attributes']):
                        self._module.fail_json(msg='\nPlaybook aborted: The route map "set"community "none" attribute is configured.\n\nPlease remove the conflicting configuration to configure other community "set" attributes.\n')
                comm_attr_rest_name = {'local_as': 'openconfig-bgp-types:NO_EXPORT_SUBCONFED', 'no_advertise': 'openconfig-bgp-types:NO_ADVERTISE', 'no_export': 'openconfig-bgp-types:NO_EXPORT', 'no_peer': 'openconfig-bgp-types:NOPEER', 'additive': 'openconfig-routing-policy-ext:ADDITIVE'}
                for comm_attr in comm_attr_list:
                    rmap_set_communities_cfg.append(comm_attr_rest_name[comm_attr])
    if cmd_set_top.get('extcommunity'):
        route_map_bgp_actions['set-ext-community'] = {'config': {'method': 'INLINE', 'options': 'ADD'}, 'inline': {'config': {'communities': []}}}
        rmap_set_extcommunities_cfg = route_map_bgp_actions['set-ext-community']['inline']['config']['communities']
        if cmd_set_top['extcommunity'].get('rt'):
            rt_list = cmd_set_top['extcommunity']['rt']
            for rt_val in rt_list:
                rmap_set_extcommunities_cfg.append('route-target:' + rt_val)
        if cmd_set_top['extcommunity'].get('soo'):
            soo_list = cmd_set_top['extcommunity']['soo']
            for soo in soo_list:
                rmap_set_extcommunities_cfg.append('route-origin:' + soo)
    route_map_bgp_actions['config'] = {}
    route_map_bgp_actions_cfg = route_map_actions['openconfig-bgp-policy:bgp-actions']['config']
    if cmd_set_top.get('ip_next_hop'):
        route_map_bgp_actions_cfg['set-next-hop'] = cmd_set_top['ip_next_hop']
    if cmd_set_top.get('ipv6_next_hop'):
        if cmd_set_top['ipv6_next_hop'].get('global_addr'):
            route_map_bgp_actions_cfg['set-ipv6-next-hop-global'] = cmd_set_top['ipv6_next_hop']['global_addr']
        if cmd_set_top['ipv6_next_hop'].get('prefer_global') is not None:
            boolval = self.yaml_bool_to_python_bool(cmd_set_top['ipv6_next_hop']['prefer_global'])
            route_map_bgp_actions_cfg['set-ipv6-next-hop-prefer-global'] = boolval
    if cmd_set_top.get('local_preference'):
        route_map_bgp_actions_cfg['set-local-pref'] = cmd_set_top['local_preference']
    if cmd_set_top.get('metric'):
        route_map_actions['metric-action'] = {'config': {}}
        route_map_metric_actions = route_map_actions['metric-action']['config']
        if cmd_set_top['metric'].get('value'):
            route_map_metric_actions['metric'] = cmd_set_top['metric']['value']
            route_map_metric_actions['action'] = 'openconfig-routing-policy:METRIC_SET_VALUE'
            route_map_bgp_actions_cfg['set-med'] = cmd_set_top['metric']['value']
        elif cmd_set_top['metric'].get('rtt_action'):
            if cmd_set_top['metric']['rtt_action'] == 'set':
                route_map_metric_actions['action'] = 'openconfig-routing-policy:METRIC_SET_RTT'
            elif cmd_set_top['metric']['rtt_action'] == 'add':
                route_map_metric_actions['action'] = 'openconfig-routing-policy:METRIC_ADD_RTT'
            elif cmd_set_top['metric']['rtt_action'] == 'subtract':
                route_map_metric_actions['action'] = 'openconfig-routing-policy:METRIC_SUBTRACT_RTT'
        if not route_map_metric_actions:
            route_map_actions.pop('metric-action')
    if cmd_set_top.get('origin'):
        route_map_bgp_actions_cfg['set-route-origin'] = cmd_set_top['origin'].upper()
    if cmd_set_top.get('weight'):
        route_map_bgp_actions_cfg['set-weight'] = cmd_set_top['weight']