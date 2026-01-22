from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import to_request
from ansible.module_utils.connection import ConnectionError
def get_modify_max_med_requests(self, vrf_name, max_med):
    request = None
    method = PATCH
    payload = {}
    on_startup_time = max_med.get('on_startup', {}).get('timer')
    on_startup_med = max_med.get('on_startup', {}).get('med_val')
    if on_startup_med is not None:
        payload = {'max-med': {'config': {'max-med-val': on_startup_med, 'time': on_startup_time}}}
    if payload:
        url = '%s=%s/%s/global/max-med' % (self.network_instance_path, vrf_name, self.protocol_bgp_path)
        request = {'path': url, 'method': method, 'data': payload}
    return [request]