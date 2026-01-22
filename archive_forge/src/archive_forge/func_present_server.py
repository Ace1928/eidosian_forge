from __future__ import absolute_import, division, print_function
from datetime import datetime, timedelta
from time import sleep
from copy import deepcopy
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.api import (
def present_server(self):
    server_info = self._get_server_info()
    if server_info.get('state') != 'absent':
        if self._module.params.get('state') == 'stopped':
            server_info = self._start_stop_server(server_info, target_state='stopped')
        server_info = self._update_server(server_info)
        if self._module.params.get('state') == 'running':
            server_info = self._start_stop_server(server_info, target_state='running')
    else:
        server_info = self._create_server(server_info)
        server_info = self._start_stop_server(server_info, target_state=self._module.params.get('state'))
    return server_info