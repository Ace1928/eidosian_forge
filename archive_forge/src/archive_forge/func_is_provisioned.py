from __future__ import absolute_import, division, print_function
import time
from collections import namedtuple
from datetime import datetime
from ansible.module_utils.basic import (
from ipaddress import ip_interface
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.ipaddress import is_valid_ip
from ..module_utils.urls import parseStats
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def is_provisioned(self):
    uri = 'https://{0}:{1}/mgmt/tm/vcmp/guest/{2}/stats'.format(self.client.provider['server'], self.client.provider['server_port'], self.want.name)
    resp = self.client.api.get(uri)
    try:
        response = resp.json()
    except ValueError:
        return False
    if 'code' in response and response['code'] == 400:
        if 'message' in response:
            raise F5ModuleError(response['message'])
        else:
            raise F5ModuleError(resp.content)
    if resp.status == 404 or ('code' in response and response['code'] == 404):
        return False
    result = parseStats(response)
    if 'stats' in result:
        if result['stats']['requestedState'] == 'provisioned':
            if result['stats']['vmStatus'] == 'stopped':
                return True
    return False