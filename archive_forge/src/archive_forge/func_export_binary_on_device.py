from __future__ import absolute_import, division, print_function
import os
import time
import tempfile
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def export_binary_on_device(self):
    full_name = fq_name(self.want.partition, self.want.name)
    cmd = 'tmsh save asm policy {0} bin-file {1}'.format(full_name, self.want.file)
    uri = 'https://{0}:{1}/mgmt/tm/util/bash/'.format(self.client.provider['server'], self.client.provider['server_port'])
    args = dict(command='run', utilCmdArgs='-c "{0}"'.format(cmd))
    resp = self.client.api.post(uri, json=args)
    try:
        response = resp.json()
        if 'commandResult' in response:
            if 'Error' in response['commandResult'] or 'error' in response['commandResult']:
                raise F5ModuleError(response['commandResult'])
    except ValueError as ex:
        raise F5ModuleError(str(ex))
    if resp.status not in [200, 201] or ('code' in response and response['code'] not in [200, 201]):
        raise F5ModuleError(resp.content)
    self._stat_binary_on_device()
    self._move_binary_to_download()
    return True