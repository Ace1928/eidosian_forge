from __future__ import absolute_import, division, print_function
import os
import re
import tempfile
import time
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def create_async_task_on_device(self):
    if self.want.encryption_password:
        params = dict(command='save', name=self.want.src, options=[{'passphrase': self.want.encryption_password}])
    else:
        params = dict(command='save', name=self.want.src)
    uri = 'https://{0}:{1}/mgmt/tm/task/sys/ucs'.format(self.client.provider['server'], self.client.provider['server_port'])
    resp = self.client.api.post(uri, json=params)
    try:
        response = resp.json()
    except ValueError as ex:
        raise F5ModuleError(str(ex))
    if resp.status in [200, 201] or ('code' in response and response['code'] in [200, 201]):
        return response['_taskId']
    raise F5ModuleError(resp.content)