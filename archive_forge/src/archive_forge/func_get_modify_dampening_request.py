from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.bgp_utils import (
from ansible.module_utils.connection import ConnectionError
def get_modify_dampening_request(self, vrf_name, conf_afi, conf_safi, conf_dampening):
    request = None
    afi_safi = ('%s_%s' % (conf_afi, conf_safi)).upper()
    url = '%s=%s/%s/' % (self.network_instance_path, vrf_name, self.protocol_bgp_path)
    url += '%s=%s/route-flap-damping' % (self.afi_safi_path, afi_safi)
    damp_payload = {'route-flap-damping': {'config': {'enabled': conf_dampening}}}
    if damp_payload:
        request = {'path': url, 'method': PATCH, 'data': damp_payload}
    return request