from __future__ import absolute_import, division, print_function
import os
import re
import time
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def wait_for_configuration_reload(self):
    noops = 0
    while noops < 4:
        time.sleep(3)
        try:
            params = dict(command='run', utilCmdArgs='-c "tmsh show sys mcp-state"')
            uri = 'https://{0}:{1}/mgmt/tm/util/bash'.format(self.client.provider['server'], self.client.provider['server_port'])
            resp = self.client.api.post(uri, json=params)
            try:
                output = resp.json()
            except ValueError as ex:
                raise F5ModuleError(str(ex))
            if 'code' in output and output['code'] in [400, 403]:
                if 'message' in output:
                    raise F5ModuleError(output['message'])
                else:
                    raise F5ModuleError(resp.content)
        except Exception:
            continue
        if 'commandResult' not in output:
            continue
        result = output['commandResult']
        if self._is_config_reloading_failed_on_device(result):
            raise F5ModuleError('Failed to reload the configuration. This may be due to a cross-version incompatibility. {0}'.format(result))
        if self._is_config_reloading_success_on_device(result):
            if self._is_config_reloading_running_on_device(result):
                noops += 1
                continue
        noops = 0