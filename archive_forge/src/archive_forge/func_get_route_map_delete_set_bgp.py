from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
def get_route_map_delete_set_bgp(self, command, set_both_keys, cmd_rmap_have, requests):
    """Append to the input list of REST API requests the REST APIs needed
        for deletion of all eligible "set" attributes defined within the
        BGP "set" conditions section of the openconfig routing-policy
        definitions for "policy-definitions" (route maps)."""
    cmd_set_top = command['set']
    cfg_set_top = cmd_rmap_have.get('set')
    conf_map_name = command['map_name']
    conf_seq_num = command['sequence_num']
    req_seq_num = str(conf_seq_num)
    bgp_set_delete_req_base = self.route_map_stmt_base_uri.format(conf_map_name, req_seq_num) + 'actions/openconfig-bgp-policy:bgp-actions/'
    self.get_route_map_delete_set_bgp_cfg(command, set_both_keys, cmd_rmap_have, requests)
    if 'as_path_prepend' in set_both_keys and cmd_set_top['as_path_prepend'] == cfg_set_top['as_path_prepend']:
        request_uri = bgp_set_delete_req_base + 'set-as-path-prepend'
        request = {'path': request_uri, 'method': DELETE}
        requests.append(request)
    elif cmd_set_top.get('as_path_prepend'):
        cmd_set_top.pop('as_path_prepend')
        if not cmd_set_top:
            return
    if 'comm_list_delete' in set_both_keys and cmd_set_top['comm_list_delete'] == cfg_set_top['comm_list_delete']:
        request_uri = bgp_set_delete_req_base + 'set-community-delete'
        request = {'path': request_uri, 'method': DELETE}
        requests.append(request)
    elif cmd_set_top.get('comm_list_delete'):
        cmd_set_top.pop('comm_list_delete')
        if not cmd_set_top:
            return
    if 'community' not in set_both_keys:
        if cmd_set_top.get('community'):
            cmd_set_top.pop('community')
            if not cmd_set_top:
                return
    else:
        community_attr_remove_list = []
        set_community_delete_attrs = []
        if cmd_set_top['community'].get('community_attributes'):
            if cfg_set_top['community'].get('community_attributes'):
                for community_attr in cmd_set_top['community']['community_attributes']:
                    if community_attr in cfg_set_top['community']['community_attributes']:
                        community_rest_name = self.set_community_rest_names[community_attr]
                        set_community_delete_attrs.append(community_rest_name)
                    else:
                        community_attr_remove_list.append(community_attr)
                for community_attr in community_attr_remove_list:
                    cmd_set_top['community']['community_attributes'].remove(community_attr)
                if not cmd_set_top['community']['community_attributes']:
                    cmd_set_top['community'].pop('community_attributes')
            else:
                cmd_set_top['community'].pop('community_attributes')
            if not cmd_set_top['community']:
                cmd_set_top.pop('community')
                if not cmd_set_top:
                    return
        if cmd_set_top.get('community') and cmd_set_top['community'].get('community_number'):
            community_number_remove_list = []
            if cfg_set_top['community'].get('community_number'):
                for community_number in cmd_set_top['community']['community_number']:
                    if community_number in cfg_set_top['community']['community_number']:
                        set_community_delete_attrs.append(community_number)
                    else:
                        community_number_remove_list.append(community_number)
                for community_number in community_number_remove_list:
                    cmd_set_top['community']['community_number'].remove(community_number)
                if not cmd_set_top['community']['community_number']:
                    cmd_set_top['community'].pop('community_number')
            else:
                cmd_set_top['community'].pop('community_number')
            if not cmd_set_top['community']:
                cmd_set_top.pop('community')
                if not cmd_set_top:
                    return
        if set_community_delete_attrs:
            bgp_set_delete_community_uri = bgp_set_delete_req_base + 'set-community'
            bgp_set_delete_comm_payload = {'openconfig-bgp-policy:set-community': {}}
            bgp_set_delete_comm_payload_contents = bgp_set_delete_comm_payload['openconfig-bgp-policy:set-community']
            bgp_set_delete_comm_payload_contents['config'] = {'method': 'INLINE', 'options': 'REMOVE'}
            bgp_set_delete_comm_payload_contents['inline'] = {'config': {'communities': set_community_delete_attrs}}
            request = {'path': bgp_set_delete_community_uri, 'method': PATCH, 'data': bgp_set_delete_comm_payload}
            requests.append(request)
    if 'extcommunity' not in set_both_keys:
        if cmd_set_top.get('extcommunity'):
            cmd_set_top.pop('extcommunity')
            if not cmd_set_top:
                return
    else:
        set_extcommunity_delete_attrs = []
        for extcomm_type in self.set_extcomm_rest_names:
            ext_comm_number_remove_list = []
            if cmd_set_top['extcommunity'].get(extcomm_type):
                if cfg_set_top['extcommunity'].get(extcomm_type):
                    for extcomm_number in cmd_set_top['extcommunity'][extcomm_type]:
                        if extcomm_number in cfg_set_top['extcommunity'][extcomm_type]:
                            set_extcommunity_delete_attrs.append(self.set_extcomm_rest_names[extcomm_type] + extcomm_number)
                        else:
                            ext_comm_number_remove_list.append(extcomm_number)
                    for extcomm_number in ext_comm_number_remove_list:
                        cmd_set_top['extcommunity'][extcomm_type].remove(extcomm_number)
                    if not cmd_set_top['extcommunity'][extcomm_type]:
                        cmd_set_top['extcommunity'].pop(extcomm_type)
                else:
                    cmd_set_top['extcommunity'].pop(extcomm_type)
                if not cmd_set_top['extcommunity']:
                    cmd_set_top.pop('extcommunity')
                    if not cmd_set_top:
                        return
        if set_extcommunity_delete_attrs:
            bgp_set_delete_extcomm_uri = bgp_set_delete_req_base + 'set-ext-community'
            bgp_set_delete_extcomm_payload = {'openconfig-bgp-policy:set-ext-community': {}}
            bgp_set_delete_comm_payload_contents = bgp_set_delete_extcomm_payload['openconfig-bgp-policy:set-ext-community']
            bgp_set_delete_comm_payload_contents['config'] = {'method': 'INLINE', 'options': 'REMOVE'}
            bgp_set_delete_comm_payload_contents['inline'] = {'config': {'communities': set_extcommunity_delete_attrs}}
            request = {'path': bgp_set_delete_extcomm_uri, 'method': PATCH, 'data': bgp_set_delete_extcomm_payload}
            requests.append(request)