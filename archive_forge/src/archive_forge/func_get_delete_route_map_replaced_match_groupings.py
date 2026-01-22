from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
def get_delete_route_map_replaced_match_groupings(self, command, cmd_rmap_have, requests):
    """For the route map specified by the input "command", create requests
        to delete any existing route map "match" configuration groupings for which
        modified attribute requests are specified"""
    if not command.get('match'):
        return
    conf_map_name = command.get('map_name', None)
    conf_seq_num = command.get('sequence_num', None)
    req_seq_num = str(conf_seq_num)
    cmd_match_top = command['match']
    cfg_match_top = cmd_rmap_have.get('match')
    if not cfg_match_top:
        command.pop('match')
        return
    match_delete_req_base = self.route_map_stmt_base_uri.format(conf_map_name, req_seq_num) + 'conditions/'
    cmd_match_keys = cmd_match_top.keys()
    cfg_match_keys = cfg_match_top.keys()
    peer_str = ''
    if 'peer' in cfg_match_keys:
        peer_dict = cfg_match_top['peer']
        peer_key = list(peer_dict.keys())[0]
        peer_str = peer_dict[peer_key]
    bgp_match_delete_req_base = match_delete_req_base + 'openconfig-bgp-policy:bgp-conditions/'
    match_top_level_keys = ['as_path', 'community', 'ext_comm', 'interface', 'ipv6', 'local_preference', 'metric', 'origin', 'peer', 'source_protocol', 'source_vrf', 'tag']
    match_multi_level_keys = ['evpn', 'ip']
    match_uri_attr = {'as_path': bgp_match_delete_req_base + 'match-as-path-set', 'community': bgp_match_delete_req_base + 'config/community-set', 'evpn': bgp_match_delete_req_base + 'openconfig-bgp-policy-ext:match-evpn-set/config/', 'ext_comm': bgp_match_delete_req_base + 'config/ext-community-set', 'interface': match_delete_req_base + 'match-interface', 'ip': {'address': match_delete_req_base + 'match-prefix-set/config/prefix-set', 'next_hop': bgp_match_delete_req_base + 'config/openconfig-bgp-policy-ext:next-hop-set'}, 'ipv6': match_delete_req_base + 'match-prefix-set/config/openconfig-routing-policy-ext:ipv6-prefix-set', 'local_preference': bgp_match_delete_req_base + 'config/local-pref-eq', 'metric': bgp_match_delete_req_base + 'config/med-eq', 'origin': bgp_match_delete_req_base + 'config/origin-eq', 'peer': match_delete_req_base + 'match-neighbor-set/config/openconfig-routing-policy-ext:address={0}'.format(peer_str), 'source_protocol': match_delete_req_base + 'config/install-protocol-eq', 'source_vrf': match_delete_req_base + 'openconfig-routing-policy-ext:match-src-network-instance', 'tag': match_delete_req_base + 'match-tag-set/config/openconfig-routing-policy-ext:tag-value'}
    cfg_top_level_key_set = set(cfg_match_keys).intersection(set(match_top_level_keys))
    cmd_top_level_key_set = set(cmd_match_keys).intersection(set(match_top_level_keys))
    symmetric_diff_set = cmd_top_level_key_set.symmetric_difference(cfg_top_level_key_set)
    intersection_diff_set = cmd_top_level_key_set.intersection(cfg_top_level_key_set)
    cmd_delete_dict = {}
    if cmd_top_level_key_set and symmetric_diff_set or any((keyname for keyname in intersection_diff_set if cmd_match_top[keyname] != cfg_match_top[keyname])):
        self.delete_replaced_dict_config(cfg_key_set=cfg_top_level_key_set, cmd_key_set=cmd_top_level_key_set, cfg_parent_dict=cfg_match_top, uri_attr=match_uri_attr, uri_dict_key='cfg_dict_member_key', deletion_dict=cmd_delete_dict, requests=requests)
        match_dict_deletions = {}
        for match_key in match_multi_level_keys:
            cfg_key_set = {}
            cmd_key_set = {}
            if match_key in cfg_match_top:
                cfg_key_set = set(cfg_match_top[match_key].keys())
                if match_key in cfg_match_top:
                    cmd_key_set = []
                    if cmd_match_top.get(match_key):
                        cmd_key_set = set(cmd_match_top[match_key].keys())
                match_dict_deletions[match_key] = {}
                match_dict_deletions_subdict = match_dict_deletions[match_key]
                self.delete_replaced_dict_config(cfg_key_set=cfg_key_set, cmd_key_set=cmd_key_set, cfg_parent_dict=cfg_match_top[match_key], uri_attr=match_uri_attr, uri_dict_key=match_key, deletion_dict=match_dict_deletions_subdict, requests=requests)
        command.pop('match')
        if cmd_delete_dict:
            command['match'] = cmd_delete_dict
            command['match'].update(match_dict_deletions)
        return
    match_key_deletions = {}
    for match_key in match_multi_level_keys:
        if match_key in cmd_match_top:
            if match_key in cfg_match_top:
                cmd_key_set = set(cmd_match_top[match_key].keys())
                cfg_key_set = set(cfg_match_top[match_key].keys())
                symmetric_diff_set = cmd_key_set.symmetric_difference(cfg_key_set)
                intersection_diff_set = cmd_key_set.intersection(cfg_key_set)
                if symmetric_diff_set or any((keyname for keyname in intersection_diff_set if cmd_match_top[match_key][keyname] != cfg_match_top[match_key][keyname])):
                    match_key_deletions[match_key] = {}
                    match_key_deletions_subdict = match_key_deletions[match_key]
                    self.delete_replaced_dict_config(cfg_key_set=cfg_key_set, cmd_key_set=cmd_key_set, cfg_parent_dict=cfg_match_top[match_key], uri_attr=match_uri_attr, uri_dict_key=match_key, deletion_dict=match_key_deletions_subdict, requests=requests)
    command.pop('match')
    if match_key_deletions:
        command['match'] = match_key_deletions