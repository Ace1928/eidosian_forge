from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.bgp_utils import (
from ansible.module_utils.connection import ConnectionError
def get_delete_max_path_requests(self, vrf_name, conf_afi, conf_safi, conf_max_path, is_delete_all, mat_max_path):
    requests = []
    afi_safi = ('%s_%s' % (conf_afi, conf_safi)).upper()
    url = '%s=%s/%s/' % (self.network_instance_path, vrf_name, self.protocol_bgp_path)
    url += '%s=%s/use-multiple-paths/' % (self.afi_safi_path, afi_safi)
    conf_ebgp = conf_max_path.get('ebgp', None)
    conf_ibgp = conf_max_path.get('ibgp', None)
    mat_ebgp = None
    mat_ibgp = None
    if mat_max_path:
        mat_ebgp = mat_max_path.get('ebgp', None)
        mat_ibgp = mat_max_path.get('ibgp', None)
    if conf_ebgp and mat_ebgp and (mat_ebgp != 1) or (is_delete_all and conf_ebgp != 1):
        requests.append({'path': url + 'ebgp/config/maximum-paths', 'method': DELETE})
    if conf_ibgp and mat_ibgp and (mat_ibgp != 1) or (is_delete_all and conf_ibgp != 1):
        requests.append({'path': url + 'ibgp/config/maximum-paths', 'method': DELETE})
    return requests