from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
def build_create_aaa_payload(self, commands):
    payload = {}
    auth_method_list = []
    if 'authentication' in commands and commands['authentication']:
        payload = {'openconfig-system:aaa': {'authentication': {'config': {}}}}
        if 'local' in commands['authentication']['data'] and commands['authentication']['data']['local']:
            auth_method_list.append('local')
        if 'group' in commands['authentication']['data'] and commands['authentication']['data']['group']:
            auth_method = commands['authentication']['data']['group']
            auth_method_list.append(auth_method)
        if auth_method_list:
            cfg = {'authentication-method': auth_method_list}
            payload['openconfig-system:aaa']['authentication']['config'].update(cfg)
        if 'fail_through' in commands['authentication']['data']:
            cfg = {'failthrough': str(commands['authentication']['data']['fail_through'])}
            payload['openconfig-system:aaa']['authentication']['config'].update(cfg)
    return payload