from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
def get_delete_one_map_replaced_groupings(self, command, have, requests):
    """For the route map specified by the input "command", create requests
        to delete any existing route map configuration groupings for which
        modified attribute requests are specified"""
    if not command:
        return {}
    conf_map_name = command.get('map_name', None)
    conf_seq_num = command.get('sequence_num', None)
    if not conf_map_name or not conf_seq_num:
        return {}
    cmd_rmap_have = self.get_matching_map(conf_map_name, conf_seq_num, have)
    if not cmd_rmap_have:
        command = {}
        return command
    self.get_delete_route_map_replaced_match_groupings(command, cmd_rmap_have, requests)
    replaced_set_group_requests = []
    self.get_delete_route_map_replaced_set_groupings(command, cmd_rmap_have, replaced_set_group_requests)
    if replaced_set_group_requests:
        requests.extend(replaced_set_group_requests)
    return command