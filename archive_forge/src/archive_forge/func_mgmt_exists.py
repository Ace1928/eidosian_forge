from __future__ import absolute_import, division, print_function
import time
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def mgmt_exists(self):
    uri = 'https://{0}:{1}/mgmt/tm/sys/db/provision.extramb/'.format(self.client.provider['server'], self.client.provider['server_port'])
    resp = self.client.api.get(uri)
    try:
        response = resp.json()
    except ValueError as ex:
        raise F5ModuleError(str(ex))
    if resp.status == 404 or ('code' in response and response['code'] == 404):
        return False
    if int(response['value']) != 0 and self.want.memory == 0:
        return False
    if int(response['value']) == 0 and self.want.memory == 0:
        return True
    if int(response['value']) == self.want.memory:
        return True
    return False