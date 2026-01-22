from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible.module_utils.connection import ConnectionError
def get_delete_primary_ip_request(self, conf, matched, name, del_primary_ip):
    requests = []
    url = 'data/sonic-vxlan:sonic-vxlan/VXLAN_TUNNEL/VXLAN_TUNNEL_LIST={name}/primary_ip'
    is_change_needed = False
    if matched:
        matched_primary_ip = matched.get('primary_ip', None)
        if matched_primary_ip and matched_primary_ip == del_primary_ip:
            is_change_needed = True
    if is_change_needed:
        request = {'path': url.format(name=name), 'method': DELETE}
        requests.append(request)
    return requests