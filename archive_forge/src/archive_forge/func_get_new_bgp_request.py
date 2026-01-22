from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import to_request
from ansible.module_utils.connection import ConnectionError
def get_new_bgp_request(self, vrf_name, as_val):
    request = None
    url = None
    method = PATCH
    payload = {}
    cfg = {}
    if as_val:
        as_cfg = {'config': {'as': float(as_val)}}
        global_cfg = {'global': as_cfg}
        cfg = {'bgp': global_cfg}
        cfg['name'] = 'bgp'
        cfg['identifier'] = 'openconfig-policy-types:BGP'
    if cfg:
        payload['openconfig-network-instance:protocol'] = [cfg]
        url = '%s=%s/protocols/protocol/' % (self.network_instance_path, vrf_name)
        request = {'path': url, 'method': method, 'data': payload}
    return request