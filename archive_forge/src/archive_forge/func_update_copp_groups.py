from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.copp.copp import CoppArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
def update_copp_groups(self, data):
    config_dict = {}
    all_copp_groups = []
    if data:
        copp_groups = data.get('copp-groups', None)
        if copp_groups:
            copp_group_list = copp_groups.get('copp-group', None)
            if copp_group_list:
                for group in copp_group_list:
                    group_dict = {}
                    copp_name = group['name']
                    config = group['config']
                    trap_priority = config.get('trap-priority', None)
                    trap_action = config.get('trap-action', None)
                    queue = config.get('queue', None)
                    cir = config.get('cir', None)
                    cbs = config.get('cbs', None)
                    if copp_name:
                        group_dict['copp_name'] = copp_name
                    if trap_priority:
                        group_dict['trap_priority'] = trap_priority
                    if trap_action:
                        group_dict['trap_action'] = trap_action
                    if queue:
                        group_dict['queue'] = queue
                    if cir:
                        group_dict['cir'] = cir
                    if cbs:
                        group_dict['cbs'] = cbs
                    if group_dict:
                        all_copp_groups.append(group_dict)
    if all_copp_groups:
        config_dict['copp_groups'] = all_copp_groups
    return config_dict