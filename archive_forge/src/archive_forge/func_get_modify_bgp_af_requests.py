from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.bgp_utils import (
from ansible.module_utils.connection import ConnectionError
def get_modify_bgp_af_requests(self, commands, have):
    requests = []
    if not commands:
        return requests
    for conf in commands:
        vrf_name = conf['vrf_name']
        as_val = conf['bgp_as']
        match = next((cfg for cfg in have if cfg['vrf_name'] == vrf_name and cfg['bgp_as'] == as_val), None)
        modify_reqs = self.get_modify_requests(conf, match, vrf_name)
        if modify_reqs:
            requests.extend(modify_reqs)
    return requests