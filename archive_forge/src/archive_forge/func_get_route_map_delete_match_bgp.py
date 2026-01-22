from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
def get_route_map_delete_match_bgp(self, command, match_both_keys, cmd_rmap_have, requests):
    """Append to the input list of REST API requests the REST APIs needed
        for deletion of all eligible "match" attributes defined within the
        BGP match conditions section of the openconfig routing-policy
        definitions for "policy-definitions" (route maps)."""
    conf_map_name = command.get('map_name', None)
    conf_seq_num = command.get('sequence_num', None)
    req_seq_num = str(conf_seq_num)
    match_top = command['match']
    cfg_match_top = cmd_rmap_have.get('match')
    route_map_stmt_base_uri_fmt = self.route_map_stmt_base_uri.format(conf_map_name, req_seq_num)
    bgp_match_delete_req_base = route_map_stmt_base_uri_fmt + 'conditions/openconfig-bgp-policy:bgp-conditions/'
    self.get_route_map_delete_match_bgp_cfg(command, match_both_keys, cmd_rmap_have, requests)
    if 'as_path' in match_both_keys and match_top['as_path'] == cfg_match_top['as_path']:
        request_uri = bgp_match_delete_req_base + 'match-as-path-set'
        request = {'path': request_uri, 'method': DELETE}
        requests.append(request)
    elif match_top.get('as_path'):
        match_top.pop('as_path')
    if 'evpn' in match_both_keys:
        evpn_cfg_delete_base = bgp_match_delete_req_base + 'openconfig-bgp-policy-ext:match-evpn-set/config/'
        evpn_attrs = match_top['evpn']
        evpn_match_keys = evpn_attrs.keys()
        evpn_rest_attr = {'default_route': 'default-type5-route', 'route_type': 'route-type', 'vni': 'vni-number'}
        pop_list = []
        for key in evpn_match_keys:
            if key not in cfg_match_top['evpn'] or evpn_attrs[key] != cfg_match_top['evpn'][key]:
                pop_list.append(key)
            else:
                request_uri = evpn_cfg_delete_base + evpn_rest_attr[key]
                request = {'path': request_uri, 'method': DELETE}
                requests.append(request)
        for key in pop_list:
            match_top['evpn'].pop(key)
        if not match_top['evpn']:
            match_top.pop('evpn')