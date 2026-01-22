from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import to_request
from ansible.module_utils.connection import ConnectionError
def get_modify_route_selection_req(self, vrf_name, compare_routerid, as_path, med):
    requests = []
    if compare_routerid is None and (not as_path) and (not med):
        return requests
    route_selection_cfg = {}
    as_path_confed = None
    as_path_ignore = None
    med_confed = None
    med_missing_as_worst = None
    always_compare_med = None
    if compare_routerid is not None:
        route_selection_cfg['external-compare-router-id'] = compare_routerid
    if as_path:
        as_path_confed = as_path.get('confed', None)
        as_path_ignore = as_path.get('ignore', None)
        if as_path_confed is not None:
            route_selection_cfg['compare-confed-as-path'] = as_path_confed
        if as_path_ignore is not None:
            route_selection_cfg['ignore-as-path-length'] = as_path_ignore
    if med:
        med_confed = med.get('confed', None)
        med_missing_as_worst = med.get('missing_as_worst', None)
        always_compare_med = med.get('always_compare_med', None)
        if med_confed is not None:
            route_selection_cfg['med-confed'] = med_confed
        if med_missing_as_worst is not None:
            route_selection_cfg['med-missing-as-worst'] = med_missing_as_worst
        if always_compare_med is not None:
            route_selection_cfg['always-compare-med'] = always_compare_med
    method = PATCH
    payload = {'route-selection-options': {'config': route_selection_cfg}}
    if payload:
        url = '%s=%s/%s/global/route-selection-options' % (self.network_instance_path, vrf_name, self.protocol_bgp_path)
        request = {'path': url, 'method': method, 'data': payload}
        requests.append(request)
    return requests