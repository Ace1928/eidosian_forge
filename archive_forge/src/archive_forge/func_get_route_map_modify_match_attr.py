from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
def get_route_map_modify_match_attr(self, command, route_map_statement):
    """In the dict specified by the input route_map_statement paramenter,
        provide REST API definitions of all "match" attributes contained in the
        user input command dict specified by the "command" input parameter
        to this function."""
    match_top = command.get('match')
    if not match_top:
        return
    route_map_statement['conditions'] = {}
    route_map_statement['conditions']['openconfig-bgp-policy:bgp-conditions'] = {}
    route_map_match_bgp_policy = route_map_statement['conditions']['openconfig-bgp-policy:bgp-conditions']
    if match_top.get('as_path'):
        route_map_match_bgp_policy['match-as-path-set'] = {'config': {'as-path-set': match_top['as_path'], 'match-set-options': 'ANY'}}
    if match_top.get('evpn'):
        route_map_match_bgp_policy['openconfig-policy-ext:match-evpn-set'] = {'config': {}}
        route_map_match_bgp_evpn = route_map_match_bgp_policy['openconfig-policy-ext:match-evpn-set']['config']
        if match_top['evpn'].get('default_route') is not None:
            boolval = self.yaml_bool_to_python_bool(match_top['evpn']['default_route'])
            route_map_match_bgp_evpn['default-type5-route'] = boolval
        if match_top['evpn'].get('route_type'):
            route_type_rest_name = 'openconfig-bgp-policy-ext:' + match_top['evpn']['route_type'].upper()
            route_map_match_bgp_evpn['route-type'] = route_type_rest_name
        if match_top['evpn'].get('vni'):
            route_map_match_bgp_evpn['vni-number'] = match_top['evpn']['vni']
        if not route_map_match_bgp_evpn:
            route_map_match_bgp_policy.pop('openconfig-policy-ext:match-evpn-set')
    route_map_match_bgp_policy['config'] = {}
    if match_top.get('local_preference'):
        route_map_match_bgp_policy['config']['local-pref-eq'] = match_top['local_preference']
    if match_top.get('metric'):
        route_map_match_bgp_policy['config']['med-eq'] = match_top['metric']
    if match_top.get('origin'):
        route_map_match_bgp_policy['config']['origin-eq'] = match_top['origin'].upper()
    if match_top.get('community'):
        route_map_match_bgp_policy['config']['community-set'] = match_top['community']
    if match_top.get('ext_comm'):
        route_map_match_bgp_policy['config']['ext-community-set'] = match_top['ext_comm']
    if match_top.get('ip') and match_top['ip'].get('next_hop'):
        route_map_match_bgp_policy['config']['openconfig-bgp-policy-ext:next-hop-set'] = match_top['ip']['next_hop']
    if not route_map_match_bgp_policy['config']:
        route_map_match_bgp_policy.pop('config')
    if not route_map_match_bgp_policy:
        route_map_statement['conditions'].pop('openconfig-bgp-policy:bgp-conditions')
    if match_top.get('interface'):
        route_map_statement['conditions']['match-interface'] = {'config': {'interface': match_top['interface']}}
    if match_top.get('ip') and match_top['ip'].get('address'):
        route_map_statement['conditions']['match-prefix-set'] = {'config': {'prefix-set': match_top['ip']['address'], 'match-set-options': 'ANY'}}
    if match_top.get('ipv6') and match_top['ipv6'].get('address'):
        if not route_map_statement['conditions'].get('match-prefix-set'):
            route_map_statement['conditions']['match-prefix-set'] = {'config': {'openconfig-routing-policy-ext:ipv6-prefix-set': match_top['ipv6']['address'], 'match-set-options': 'ANY'}}
        else:
            route_map_statement['conditions']['match-prefix-set']['config']['openconfig-routing-policy-ext:ipv6-prefix-set'] = match_top['ipv6']['address']
    if match_top.get('peer'):
        peer_list = list(match_top['peer'].values())
        route_map_statement['conditions']['match-neighbor-set'] = {'config': {'openconfig-routing-policy-ext:address': peer_list}}
    if match_top.get('source_protocol'):
        rest_protocol_name = ''
        if match_top['source_protocol'] in ('bgp', 'ospf', 'static'):
            rest_protocol_name = 'openconfig-policy-types:' + match_top['source_protocol'].upper()
        elif match_top['source_protocol'] == 'connected':
            rest_protocol_name = 'openconfig-policy-types:DIRECTLY_CONNECTED'
        route_map_statement['conditions']['config'] = {'install-protocol-eq': rest_protocol_name}
    if match_top.get('source_vrf'):
        route_map_statement['conditions']['openconfig-routing-policy-ext:match-src-network-instance'] = {'config': {'name': match_top['source_vrf']}}
    if match_top.get('tag'):
        route_map_statement['conditions']['match-tag-set'] = {'config': {'openconfig-routing-policy-ext:tag-value': [match_top['tag']]}}