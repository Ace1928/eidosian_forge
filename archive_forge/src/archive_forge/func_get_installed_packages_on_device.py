from __future__ import absolute_import, division, print_function
import os
import time
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.urls import urlparse
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def get_installed_packages_on_device(self):
    uri = 'https://{0}:{1}/mgmt/shared/iapp/package-management-tasks'.format(self.client.provider['server'], self.client.provider['server_port'])
    params = dict(operation='QUERY')
    resp = self.client.api.post(uri, json=params)
    try:
        response = resp.json()
    except ValueError as ex:
        raise F5ModuleError(str(ex))
    if resp.status not in [200, 201, 202] or ('code' in response and response['code'] not in [200, 201, 202]):
        raise F5ModuleError(resp.content)
    path = urlparse(response['selfLink']).path
    task = self._wait_for_task(path)
    if task['status'] == 'FINISHED':
        return task['queryResponse']
    raise F5ModuleError('Failed to find the installed packages on the device.')