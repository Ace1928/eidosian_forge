from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
def get_route_map_delete_set_bgp_cfg(self, command, set_both_keys, cmd_rmap_have, requests):
    """Append to the input list of REST API requests the REST APIs needed
        for deletion of all eligible "set" attributes defined within the
        BGP set conditions 'config' section of the openconfig routing-policy
        definitions for "policy-definitions" (route maps)."""
    cmd_set_top = command['set']
    cfg_set_top = cmd_rmap_have.get('set')
    conf_map_name = command['map_name']
    conf_seq_num = command['sequence_num']
    req_seq_num = str(conf_seq_num)
    bgp_set_delete_req_base = self.route_map_stmt_base_uri.format(conf_map_name, req_seq_num) + 'actions/openconfig-bgp-policy:bgp-actions/config/'
    bgp_cfg_keys = {'ip_next_hop', 'origin', 'local_preference', 'ipv6_next_hop', 'weight'}
    delete_bgp_keys = bgp_cfg_keys.intersection(set_both_keys)
    if not delete_bgp_keys:
        for bgp_key in bgp_cfg_keys:
            if bgp_key in cmd_set_top:
                cmd_set_top.pop(bgp_key)
        return
    delete_bgp_attrs = []
    if 'ipv6_next_hop' in delete_bgp_keys:
        delete_bgp_keys.remove('ipv6_next_hop')
        ipv6_next_hop_rest_names = {'global_addr': 'set-ipv6-next-hop-global', 'prefer_global': 'set-ipv6-next-hop-prefer-global'}
        for ipv6_next_hop_key in ipv6_next_hop_rest_names:
            if cmd_set_top['ipv6_next_hop'].get(ipv6_next_hop_key) is not None:
                if cmd_set_top['ipv6_next_hop'][ipv6_next_hop_key] == cfg_set_top['ipv6_next_hop'].get(ipv6_next_hop_key):
                    delete_bgp_attrs.append(ipv6_next_hop_rest_names[ipv6_next_hop_key])
                else:
                    cmd_set_top['ipv6_next_hop'].pop(ipv6_next_hop_key)
                    if not cmd_set_top['ipv6_next_hop']:
                        cmd_set_top.pop('ipv6_next_hop')
                        if not cmd_set_top:
                            return
        if not delete_bgp_keys and (not delete_bgp_attrs):
            return
    bgp_cfg_rest_names = {'ip_next_hop': 'set-next-hop', 'local_preference': 'set-local-pref', 'origin': 'set-route-origin', 'weight': 'set-weight'}
    for bgp_cfg_key in bgp_cfg_rest_names:
        if bgp_cfg_key in delete_bgp_keys:
            if cmd_set_top[bgp_cfg_key] == cfg_set_top[bgp_cfg_key]:
                delete_bgp_attrs.append(bgp_cfg_rest_names[bgp_cfg_key])
            else:
                cmd_set_top.pop(bgp_cfg_key)
    if not cmd_set_top:
        command.pop('set')
        return
    for delete_bgp_attr in delete_bgp_attrs:
        del_set_bgp_cfg_uri = bgp_set_delete_req_base + delete_bgp_attr
        request = {'path': del_set_bgp_cfg_uri, 'method': DELETE}
        requests.append(request)