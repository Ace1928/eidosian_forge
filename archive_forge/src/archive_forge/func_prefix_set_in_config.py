from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
def prefix_set_in_config(self, pfx_set_name, have):
    '''Determine if the prefix set specifid by "pfx_set_name" is present in
        the current switch configuration. If it is present, return the "found"
        prefix set. (Otherwise, return "None"'''
    for cfg_prefix_set in have:
        cfg_prefix_set_name = cfg_prefix_set.get('name', None)
        if cfg_prefix_set_name and cfg_prefix_set_name == pfx_set_name:
            return cfg_prefix_set
    return None