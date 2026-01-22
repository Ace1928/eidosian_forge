from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
from ansible.module_utils.connection import ConnectionError
def get_create_servers_requests(self, configs, have):
    requests = []
    method = PATCH
    url = 'data/openconfig-system:system/ntp/servers'
    server_configs = []
    for config in configs:
        if 'key_id' in config:
            config['key-id'] = config['key_id']
            config.pop('key_id')
        server_addr = config['address']
        server_config = {'address': server_addr, 'config': config}
        server_configs.append(server_config)
    payload = {'openconfig-system:servers': {'server': server_configs}}
    request = {'path': url, 'method': method, 'data': payload}
    requests.append(request)
    return requests