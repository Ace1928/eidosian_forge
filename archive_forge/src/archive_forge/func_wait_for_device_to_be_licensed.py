from __future__ import absolute_import, division, print_function
import re
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import bigiq_version
from ..module_utils.ipaddress import is_valid_ip
from ..module_utils.teem import send_teem
def wait_for_device_to_be_licensed(self):
    count = 0
    uri = 'https://{0}:{1}/mgmt/cm/device/licensing/pool/utility/licenses/{2}/offerings/{3}/members/{4}'.format(self.client.provider['server'], self.client.provider['server_port'], self.want.key, self.want.offering_id, self.want.member_id)
    while count < 3:
        resp = self.client.api.get(uri)
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if 'code' in response and response['code'] == 400:
            if 'message' in response:
                raise F5ModuleError(response['message'])
            else:
                raise F5ModuleError(resp.content)
        if response['status'] == 'LICENSED':
            count += 1
        else:
            count = 0