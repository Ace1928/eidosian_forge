from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import to_request
from ansible.module_utils.connection import ConnectionError
def get_modify_bestpath_requests(self, vrf_name, bestpath):
    requests = []
    if not bestpath:
        return requests
    compare_routerid = bestpath.get('compare_routerid', None)
    as_path = bestpath.get('as_path', None)
    med = bestpath.get('med', None)
    route_selection_req = self.get_modify_route_selection_req(vrf_name, compare_routerid, as_path, med)
    if route_selection_req:
        requests.extend(route_selection_req)
    multi_paths_req = self.get_modify_multi_paths_req(vrf_name, as_path)
    if multi_paths_req:
        requests.append(multi_paths_req)
    return requests