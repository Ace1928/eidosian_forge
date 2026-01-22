from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import to_request
from ansible.module_utils.connection import ConnectionError
def get_delete_bgp_requests(self, commands, have, is_delete_all):
    requests = []
    if is_delete_all:
        requests = self.get_delete_all_bgp_requests(commands)
    else:
        for cmd in commands:
            vrf_name = cmd['vrf_name']
            as_val = cmd['bgp_as']
            match = next((cfg for cfg in have if cfg['vrf_name'] == vrf_name and cfg['bgp_as'] == as_val), None)
            if not match:
                continue
            if cmd.get('router_id', None) or cmd.get('log_neighbor_changes', None) or cmd.get('bestpath', None) or cmd.get('rt_delay', None):
                requests.extend(self.get_delete_specific_bgp_param_request(cmd, match))
            else:
                requests.append(self.get_delete_single_bgp_request(vrf_name))
    if requests:
        default_vrf_reqs = []
        other_vrf_reqs = []
        for req in requests:
            if '=default/' in req['path']:
                default_vrf_reqs.append(req)
            else:
                other_vrf_reqs.append(req)
        requests.clear()
        requests.extend(other_vrf_reqs)
        requests.extend(default_vrf_reqs)
    return requests