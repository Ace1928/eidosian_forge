from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.bgp_utils import (
from ansible.module_utils.connection import ConnectionError
def get_delete_network_request(self, vrf_name, conf_afi, conf_safi, conf_network, is_delete_all, mat_network):
    requests = []
    afi_safi = ('%s_%s' % (conf_afi, conf_safi)).upper()
    url = '%s=%s/%s/' % (self.network_instance_path, vrf_name, self.protocol_bgp_path)
    url += '%s=%s/network-config/network=' % (self.afi_safi_path, afi_safi)
    mat_list = []
    for conf in conf_network:
        if mat_network:
            mat_prefix = next((pre for pre in mat_network if pre == conf), None)
            if mat_prefix:
                mat_list.append(mat_prefix)
    if not is_delete_all and mat_list:
        for each in mat_list:
            tmp = each.replace('/', '%2f')
            requests.append({'path': url + tmp, 'method': DELETE})
    elif is_delete_all:
        for each in conf_network:
            tmp = each.replace('/', '%2f')
            requests.append({'path': url + tmp, 'method': DELETE})
    return requests