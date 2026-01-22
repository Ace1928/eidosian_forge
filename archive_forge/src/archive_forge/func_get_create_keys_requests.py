from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
from ansible.module_utils.connection import ConnectionError
def get_create_keys_requests(self, configs, have):
    requests = []
    method = PATCH
    url = 'data/openconfig-system:system/ntp/ntp-keys'
    key_configs = []
    for config in configs:
        key_id = config['key_id']
        if 'key_id' in config:
            config['key-id'] = config['key_id']
            config.pop('key_id')
        if 'key_type' in config:
            config['key-type'] = config['key_type']
            config.pop('key_type')
        if 'key_value' in config:
            config['key-value'] = config['key_value']
            config.pop('key_value')
        key_config = {'key-id': key_id, 'config': config}
        key_configs.append(key_config)
    payload = {'openconfig-system:ntp-keys': {'ntp-key': key_configs}}
    request = {'path': url, 'method': method, 'data': payload}
    requests.append(request)
    return requests