from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible.module_utils.connection import ConnectionError
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
def get_delete_servers(self, command, have):
    requests = []
    url = 'data/openconfig-system:system/aaa/server-groups/server-group=RADIUS/servers/server='
    mat_hosts = []
    if have.get('servers', None) and have['servers'].get('host', None):
        mat_hosts = have['servers']['host']
    if command.get('servers', None):
        if command['servers'].get('host', None):
            hosts = command['servers']['host']
        else:
            hosts = mat_hosts
    if mat_hosts and hosts:
        for host in hosts:
            if next((m_host for m_host in mat_hosts if m_host['name'] == host['name']), None):
                requests.append({'path': url + host['name'], 'method': DELETE})
    return requests