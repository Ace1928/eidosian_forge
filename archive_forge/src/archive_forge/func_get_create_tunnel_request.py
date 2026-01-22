from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible.module_utils.connection import ConnectionError
def get_create_tunnel_request(self, configs, have):
    requests = []
    url = 'data/sonic-vxlan:sonic-vxlan/VXLAN_TUNNEL'
    for conf in configs:
        payload = self.build_create_tunnel_payload(conf)
        request = {'path': url, 'method': PATCH, 'data': payload}
        requests.append(request)
        if conf.get('evpn_nvo', None):
            requests.append(self.get_create_evpn_request(conf))
    return requests