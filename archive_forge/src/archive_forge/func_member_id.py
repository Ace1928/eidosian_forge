from __future__ import absolute_import, division, print_function
import re
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import bigiq_version
from ..module_utils.ipaddress import is_valid_ip
from ..module_utils.teem import send_teem
@property
def member_id(self):
    if self.device_is_address:
        filter = "deviceAddress+eq+'{0}...{0}'".format(self.device)
    elif self.device_is_name:
        filter = "deviceName+eq+'{0}'".format(self.device)
    elif self.device_is_id:
        filter = "deviceMachineId+eq+'{0}'".format(self.device)
    else:
        raise F5ModuleError("Unknown device format '{0}'".format(self.device))
    uri = 'https://{0}:{1}/mgmt/cm/device/licensing/pool/utility/licenses/{2}/offerings/{3}/members/?$filter={4}'.format(self.client.provider['server'], self.client.provider['server_port'], self.key, self.offering_id, filter)
    resp = self.client.api.get(uri)
    try:
        response = resp.json()
    except ValueError as ex:
        raise F5ModuleError(str(ex))
    if resp.status == 200 and response['totalItems'] == 0:
        return None
    elif 'code' in response and response['code'] == 400:
        if 'message' in response:
            raise F5ModuleError(response['message'])
        else:
            raise F5ModuleError(resp._content)
    result = response['items'][0]['id']
    return result