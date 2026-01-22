from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.bgp_utils import (
from ansible.module_utils.connection import ConnectionError
def get_delete_neighbor_af_routemaps_requests(self, vrf_name, conf_neighbor_val, afi, safi, routes):
    requests = []
    for route in routes:
        afi_safi_name = ('%s_%s' % (afi, safi)).upper()
        policy_type = 'import-policy' if 'in' == route['direction'] else 'export-policy'
        url = '%s=%s/%s/%s=%s/' % (self.network_instance_path, vrf_name, self.protocol_bgp_path, self.neighbor_path, conf_neighbor_val)
        url += '%s=%s/apply-policy/config/%s' % (self.afi_safi_path, afi_safi_name, policy_type)
        requests.append({'path': url, 'method': DELETE})
    return requests