from __future__ import absolute_import, division, print_function
from datetime import datetime
import os
import tempfile
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def merge_on_device(self, remote_path, verify=True):
    command = 'tmsh load sys config file {0} merge'.format(remote_path)
    if verify:
        command += ' verify'
    uri = 'https://{0}:{1}/mgmt/tm/util/bash'.format(self.client.provider['server'], self.client.provider['server_port'])
    args = dict(command='run', utilCmdArgs='-c "{0}"'.format(command))
    resp = self.client.api.post(uri, json=args)
    try:
        response = resp.json()
    except ValueError as ex:
        raise F5ModuleError(str(ex))
    if resp.status in [200, 201] or ('code' in response and response['code'] in [200, 201]):
        if 'commandResult' in response:
            return str(response['commandResult'])
        else:
            return
    raise F5ModuleError(resp.content)