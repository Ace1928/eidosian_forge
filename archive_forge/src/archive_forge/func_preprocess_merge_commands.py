from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
from ansible.module_utils.connection import ConnectionError
def preprocess_merge_commands(self, commands, want):
    if 'servers' in commands and commands['servers'] is not None:
        for server in commands['servers']:
            if 'minpoll' in server and 'maxpoll' not in server:
                want_server = dict()
                if 'servers' in want:
                    want_server = self.search_servers(server['address'], want['servers'])
                if want_server:
                    server['maxpoll'] = want_server['maxpoll']
                else:
                    err_msg = 'Internal error with NTP server maxpoll configuration.'
                    self._module.fail_json(msg=err_msg, code=500)
            if 'maxpoll' in server and 'minpoll' not in server:
                want_server = dict()
                if 'servers' in want:
                    want_server = self.search_servers(server['address'], want['servers'])
                if want_server:
                    server['minpoll'] = want_server['minpoll']
                else:
                    err_msg = 'Internal error with NTP server minpoll configuration.'
                    self._module.fail_json(msg=err_msg, code=500)