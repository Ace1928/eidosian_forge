from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
def filter_copp_groups(self, commands):
    cfg_dict = {}
    if commands:
        copp_groups = commands.get('copp_groups', None)
        if copp_groups:
            copp_groups_list = []
            for group in copp_groups:
                copp_name = group.get('copp_name', None)
                if copp_name not in reserved_copp_names:
                    copp_groups_list.append(group)
            if copp_groups_list:
                cfg_dict['copp_groups'] = copp_groups_list
    return cfg_dict