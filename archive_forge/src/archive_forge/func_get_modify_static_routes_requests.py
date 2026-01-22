from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
def get_modify_static_routes_requests(self, commands):
    requests = []
    if not commands:
        return requests
    for conf in commands:
        vrf_name = conf.get('vrf_name', None)
        static_list = conf.get('static_list', [])
        for static in static_list:
            prefix = static.get('prefix', None)
            next_hops = static.get('next_hops', [])
            if next_hops:
                for next_hop in next_hops:
                    requests.append(self.get_modify_static_route_request(vrf_name, prefix, next_hop))
    return requests