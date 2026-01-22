from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
from ansible.module_utils.connection import ConnectionError
def preprocess_want(self, want, state):
    if 'source_interfaces' in want:
        want['source_interfaces'] = normalize_interface_name_list(want['source_interfaces'], self._module)
    if state == 'deleted':
        enable_auth_want = want.get('enable_ntp_auth', None)
        if enable_auth_want is not None:
            want['enable_ntp_auth'] = True
    elif state == 'merged':
        if 'servers' in want and want['servers'] is not None:
            for server in want['servers']:
                if 'key_id' in server and (not server['key_id']):
                    server.pop('key_id')
                if 'minpoll' in server and (not server['minpoll']):
                    server.pop('minpoll')
                if 'maxpoll' in server and (not server['maxpoll']):
                    server.pop('maxpoll')
                if 'prefer' in server and server['prefer'] is None:
                    server.pop('prefer')
    if state == 'replaced' or state == 'overridden':
        enable_auth_want = want.get('enable_ntp_auth', None)
        if enable_auth_want is None:
            want['enable_ntp_auth'] = False
        if 'servers' in want and want['servers'] is not None:
            for server in want['servers']:
                if 'prefer' in server and server['prefer'] is None:
                    server['prefer'] = False