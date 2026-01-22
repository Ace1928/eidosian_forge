from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible.module_utils.connection import ConnectionError
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
def get_modify_servers_request(self, command):
    request = None
    hosts = []
    if command.get('servers', None) and command['servers'].get('host', None):
        hosts = command['servers']['host']
    if hosts:
        url = 'data/openconfig-system:system/aaa/server-groups/server-group=RADIUS/servers'
        payload = self.get_radius_server_payload(hosts)
        if payload:
            request = {'path': url, 'method': PATCH, 'data': payload}
    return request