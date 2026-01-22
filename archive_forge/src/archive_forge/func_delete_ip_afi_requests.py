from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.bgp_utils import (
from ansible.module_utils.connection import ConnectionError
def delete_ip_afi_requests(self, conf_ip_afi, mat_ip_afi, conf_afi_safi_val, url):
    requests = []
    default_policy_name = conf_ip_afi.get('default_policy_name', None)
    send_default_route = conf_ip_afi.get('send_default_route', None)
    if default_policy_name:
        self.append_delete_request(requests, default_policy_name, mat_ip_afi, 'default_policy_name', url, self.def_policy_name_path % conf_afi_safi_val)
    if send_default_route:
        self.append_delete_request(requests, send_default_route, mat_ip_afi, 'send_default_route', url, self.send_def_route_path % conf_afi_safi_val)
    return requests